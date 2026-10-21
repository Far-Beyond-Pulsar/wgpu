//! HELIO PATCH: regression test for the texture tracker's start states.
//!
//! A command buffer copies into mip 0 of a texture (so its tracker knows only
//! mip 0), then runs a compute pass that builds the rest of the mip chain:
//! each dispatch reads mip `k` and writes mip `k + 1` as a storage texture.
//! Mips 1..n-1 enter the pass as storage and leave it sampled. Merging the
//! pass into the command buffer used to record their *end* state as the
//! command buffer's start state, so the submission's transition put them in
//! the sampled layout and the pass wrote them in the wrong one (on Vulkan,
//! `SHADER_READ_ONLY_OPTIMAL` instead of `GENERAL`:
//! VUID-vkCmdDraw-None-09600). The validation layer catches that; drivers
//! that ignore layouts still produce the right values, which the test also
//! checks.

use std::borrow::Cow;

use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(MIP_CHAIN_AFTER_PARTIAL_COPY);
}

const SIZE: u32 = 8;
const MIPS: u32 = 4;

/// The largest of each 2x2 footprint of `src` into `dst`.
const SHADER_SRC: &str = "
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = id.xy * 2u;
    let a = max(textureLoad(src, p, 0).r, textureLoad(src, p + vec2(1u, 0u), 0).r);
    let b = max(textureLoad(src, p + vec2(0u, 1u), 0).r, textureLoad(src, p + vec2(1u, 1u), 0).r);
    textureStore(dst, id.xy, vec4(max(a, b)));
}
";

#[gpu_test]
static MIP_CHAIN_AFTER_PARTIAL_COPY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .limits(wgpu::Limits {
                max_storage_textures_per_shader_stage: 1,
                ..wgpu::Limits::downlevel_defaults()
            })
            // GL reads 0 back from the last mip, with or without the tracker
            // fix: a separate GL problem with single-mip views of one
            // texture bound as sampled and storage, not investigated here.
            .expect_fail(FailureCase::backend(wgpu::Backends::GL)),
    )
    .run_async(|ctx| async move {
        let device = &ctx.device;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mip chain"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let views = (0..MIPS)
            .map(|mip| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect::<Vec<_>>();

        // Mip 0's texels are 0, 1, 2, ...: the chain's last texel is the
        // largest of them.
        const ROW: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let mut upload = vec![0u8; (ROW * SIZE) as usize];
        for y in 0..SIZE {
            for x in 0..SIZE {
                let at = (y * ROW + x * 4) as usize;
                upload[at..at + 4].copy_from_slice(&((y * SIZE + x) as f32).to_le_bytes());
            }
        }
        let upload_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mip 0"),
            size: upload.len() as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&upload_buffer, 0, &upload);
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("last mip"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let bind_groups = (0..MIPS - 1)
            .map(|mip| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&views[mip as usize]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&views[mip as usize + 1]),
                        },
                    ],
                })
            })
            .collect::<Vec<_>>();
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SRC)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pl),
            entry_point: Some("main"),
            compilation_options: Default::default(),
            module: &module,
            cache: None,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Only mip 0: the command buffer's tracker learns nothing of mips 1+.
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &upload_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ROW),
                    rows_per_image: Some(SIZE),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mip chain"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            for (mip, bind_group) in bind_groups.iter().enumerate() {
                let size = SIZE >> (mip + 1);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(size, size, 1);
            }
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: MIPS - 1,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        ctx.queue.submit(Some(encoder.finish()));

        let slice = readback_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, Result::unwrap);
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        {
            let bytes = slice.get_mapped_range().unwrap();
            let value = f32::from_le_bytes(bytes[..4].try_into().unwrap());
            assert_eq!(value, (SIZE * SIZE - 1) as f32);
        }
        readback_buffer.unmap();
    });
