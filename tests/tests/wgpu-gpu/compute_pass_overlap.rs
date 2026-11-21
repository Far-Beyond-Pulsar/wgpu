//! HELIO PATCH: a compute pass's overlapping spans
//! (`ComputePass::begin_overlap` / `end_overlap`).

use wgpu_test::{
    gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([OVERLAP_SPAN, OVERLAP_SPAN_ERRORS]);
}

/// The span's indirect dispatches: more than the validation's slots, so the
/// last ones are validated one by one, as outside a span.
const INDIRECT: u32 = 300;
/// The dispatch whose arguments exceed the device's limit: discarded.
const REFUSED: u32 = 7;
/// What the direct dispatch of the span writes, after the indirect ones.
const DIRECT: u32 = 5;
const MAX_WORKGROUPS: u32 = 10;

fn parameters() -> TestParameters {
    TestParameters::default()
        .features(wgpu::Features::IMMEDIATES)
        .downlevel_flags(
            wgpu::DownlevelFlags::COMPUTE_SHADERS | wgpu::DownlevelFlags::INDIRECT_EXECUTION,
        )
        .limits(wgpu::Limits {
            max_compute_workgroups_per_dimension: MAX_WORKGROUPS,
            max_immediate_size: 4,
            ..wgpu::Limits::downlevel_defaults()
        })
}

fn workgroups(k: u32) -> u32 {
    if k == REFUSED {
        MAX_WORKGROUPS + 1
    } else {
        k % 3 + 1
    }
}

/// Each dispatch of a span writes its own element from its own arguments
/// (each validated into its own slot: one refused, the ones past the slots
/// validated one by one), and the dispatch after the span reads every write
/// of it, through a buffer that stays read-write the whole pass.
#[gpu_test]
static OVERLAP_SPAN: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_async(|ctx| async move {
        let resources = Resources::new(&ctx);

        let arguments: Vec<u32> = (0..INDIRECT).flat_map(|k| [workgroups(k), 1, 1]).collect();
        let indirect_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(INDIRECT) * 12,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&indirect_buffer, 0, bytemuck::cast_slice(&arguments));

        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&resources.write);
            pass.set_bind_group(0, &resources.bind_group, &[]);
            pass.begin_overlap();
            for k in 0..INDIRECT {
                pass.set_immediates(0, bytemuck::bytes_of(&k));
                pass.dispatch_workgroups_indirect(&indirect_buffer, u64::from(k) * 12);
            }
            pass.set_immediates(0, bytemuck::bytes_of(&INDIRECT));
            pass.dispatch_workgroups(DIRECT, 1, 1);
            pass.end_overlap();
            pass.set_pipeline(&resources.sum);
            pass.set_bind_group(0, &resources.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &resources.out_buffer,
            0,
            &resources.readback_buffer,
            0,
            resources.out_buffer.size(),
        );
        ctx.queue.submit(Some(encoder.finish()));

        resources
            .readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        let view = resources
            .readback_buffer
            .slice(..)
            .get_mapped_range()
            .unwrap();
        let out: Vec<u32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        resources.readback_buffer.unmap();

        let mut expected: Vec<u32> = (0..INDIRECT)
            .map(|k| if k == REFUSED { 0 } else { workgroups(k) })
            .collect();
        expected.push(DIRECT);
        expected.push(expected.iter().sum());
        assert_eq!(out, expected);
    });

/// Spans do not nest, and a span is ended only once.
#[gpu_test]
static OVERLAP_SPAN_ERRORS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters().enable_noop())
    .run_async(|ctx| async move {
        let resources = Resources::new(&ctx);
        for (nested, text) in [
            (true, "begin_overlap called inside an overlapping span"),
            (false, "end_overlap called outside an overlapping span"),
        ] {
            let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&resources.write);
                pass.set_bind_group(0, &resources.bind_group, &[]);
                pass.begin_overlap();
                if nested {
                    pass.begin_overlap();
                }
                pass.end_overlap();
                if !nested {
                    pass.end_overlap();
                }
            }
            ctx.queue.submit(Some(encoder.finish()));
            let error = pollster::block_on(scope.pop());
            assert!(
                error
                    .as_ref()
                    .is_some_and(|error| format!("{error}").contains(text)),
                "expected \"{text}\", got {error:?}"
            );
        }
    });

struct Resources {
    write: wgpu::ComputePipeline,
    sum: wgpu::ComputePipeline,
    out_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl Resources {
    fn new(ctx: &TestingContext) -> Self {
        // `out`: an element per indirect dispatch, the direct dispatch's,
        // then the sum of those.
        let source = format!(
            "
            const WRITES: u32 = {writes}u;

            struct Element {{
                index: u32,
            }}
            var<immediate> element: Element;

            @group(0) @binding(0)
            var<storage, read_write> out: array<u32>;

            @compute @workgroup_size(1)
            fn write(@builtin(num_workgroups) num_workgroups: vec3u, @builtin(workgroup_id) workgroup_id: vec3u) {{
                if (all(workgroup_id == vec3u())) {{
                    out[element.index] = num_workgroups.x;
                }}
            }}

            @compute @workgroup_size(1)
            fn sum() {{
                var total = 0u;
                for (var i = 0u; i < WRITES; i++) {{
                    total += out[i];
                }}
                out[WRITES] = total;
            }}
        ",
            writes = INDIRECT + 1
        );
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 4,
            });
        let pipeline = |entry_point| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let size = u64::from(INDIRECT + 2) * 4;
        let out_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: out_buffer.as_entire_binding(),
            }],
        });
        Self {
            write: pipeline("write"),
            sum: pipeline("sum"),
            out_buffer,
            readback_buffer,
            bind_group,
        }
    }
}
