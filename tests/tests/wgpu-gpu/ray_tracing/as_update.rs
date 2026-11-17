//! Acceleration structure updates (refits), forced rebuilds, and compaction racing submits.
//!
//! Every update test traces two rays along +z, from `A` and from `B`, and checks which hit, so a
//! stale structure fails. Whether a build was encoded as an update is shown by submitting it
//! after a build with different inputs: only an update is rejected then.

use std::{
    mem,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    time::Duration,
};

use crate::ray_tracing::acceleration_structure_limits;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;
use wgpu_test::{
    fail, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        BLAS_UPDATE,
        TLAS_UPDATE,
        BLAS_INCOMPATIBLE_BUILD,
        BLAS_REQUESTED_REBUILD,
        PREFER_UPDATE_WITHOUT_ALLOW_UPDATE,
        COMPACT_WHILE_SUBMITTING,
    ]);
}

/// The ray origins: `A` at x = 0 and `B` at x = 10.
const ORIGINS: [[f32; 4]; 2] = [[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]];
const HIT_A: [u32; 2] = [1, 0];
const HIT_B: [u32; 2] = [0, 1];

const IDENTITY: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

/// A transform moving geometry by `x` along the x axis.
fn translation(x: f32) -> [f32; 12] {
    [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}

/// A triangle in the plane z = 1 around (x, 0, 1).
fn triangle(x: f32) -> [[f32; 3]; 3] {
    [[x - 1.0, -1.0, 1.0], [x + 1.0, -1.0, 1.0], [x, 1.0, 1.0]]
}

/// Room for three triangles; builds use the first `vertex_count` vertices.
const MAX_VERTICES: u32 = 9;

fn size_descriptor(vertex_count: u32) -> BlasTriangleGeometrySizeDescriptor {
    BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count,
        index_format: None,
        index_count: None,
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    }
}

struct UpdateScene {
    vertices: Buffer,
    blas: Blas,
    tlas: Tlas,
    pipeline: ComputePipeline,
    origins: Buffer,
    hits: Buffer,
    readback: Buffer,
}

impl UpdateScene {
    fn new(
        ctx: &TestingContext,
        blas_mode: AccelerationStructureUpdateMode,
        tlas_mode: AccelerationStructureUpdateMode,
    ) -> Self {
        let flags = |mode| match mode {
            AccelerationStructureUpdateMode::Build => AccelerationStructureFlags::PREFER_FAST_TRACE,
            AccelerationStructureUpdateMode::PreferUpdate => {
                AccelerationStructureFlags::PREFER_FAST_TRACE
                    | AccelerationStructureFlags::ALLOW_UPDATE
            }
        };

        let vertices = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("vertices"),
            size: (MAX_VERTICES as usize * mem::size_of::<[f32; 3]>()) as BufferAddress,
            usage: BufferUsages::BLAS_INPUT | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let blas = ctx.device.create_blas(
            &CreateBlasDescriptor {
                label: Some("BLAS"),
                flags: flags(blas_mode),
                update_mode: blas_mode,
            },
            BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![size_descriptor(MAX_VERTICES)],
            },
        );

        let mut tlas = ctx.device.create_tlas(&CreateTlasDescriptor {
            label: Some("TLAS"),
            max_instances: 2,
            flags: flags(tlas_mode),
            update_mode: tlas_mode,
        });
        tlas[0] = Some(TlasInstance::new(&blas, IDENTITY, 0, 0xFF));

        let shader = ctx
            .device
            .create_shader_module(include_wgsl!("as_update.wgsl"));
        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("trace"),
                layout: None,
                module: &shader,
                entry_point: Some("trace"),
                compilation_options: Default::default(),
                cache: None,
            });

        let origins = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("origins"),
            contents: bytemuck::cast_slice(&ORIGINS),
            usage: BufferUsages::STORAGE,
        });
        let hits_size = mem::size_of::<[u32; 2]>() as BufferAddress;
        let hits = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("hits"),
            size: hits_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("hits readback"),
            size: hits_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            vertices,
            blas,
            tlas,
            pipeline,
            origins,
            hits,
            readback,
        }
    }

    /// Writes `triangles` as the vertex data.
    fn write_triangles(&self, ctx: &TestingContext, triangles: &[[[f32; 3]; 3]]) {
        ctx.queue
            .write_buffer(&self.vertices, 0, bytemuck::cast_slice(triangles));
    }

    /// Encodes a build of the BLAS from its first `vertex_count` vertices, and of the TLAS.
    fn encode_build(&self, ctx: &TestingContext, vertex_count: u32) -> CommandBuffer {
        let size = size_descriptor(vertex_count);
        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("build"),
            });
        encoder.build_acceleration_structures(
            [&BlasBuildEntry {
                blas: &self.blas,
                geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                    size: &size,
                    vertex_buffer: &self.vertices,
                    first_vertex: 0,
                    vertex_stride: mem::size_of::<[f32; 3]>() as BufferAddress,
                    index_buffer: None,
                    first_index: None,
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            }],
            [&self.tlas],
        );
        encoder.finish()
    }

    /// Encodes a build of the TLAS alone.
    fn encode_tlas_build(&self, ctx: &TestingContext) -> CommandBuffer {
        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("TLAS build"),
            });
        encoder.build_acceleration_structures([], [&self.tlas]);
        encoder.finish()
    }

    /// Traces a ray from each of [`ORIGINS`] and returns which hit.
    fn trace(&self, ctx: &TestingContext) -> [u32; 2] {
        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::AccelerationStructure(&self.tlas),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.origins.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.hits.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(ORIGINS.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.hits, 0, &self.readback, 0, None);
        ctx.queue.submit([encoder.finish()]);

        self.readback.slice(..).map_async(MapMode::Read, |_| ());
        ctx.device.poll(PollType::wait_indefinitely()).unwrap();
        let hits = {
            let view = self.readback.slice(..).get_mapped_range().unwrap();
            let hits: &[u32] = bytemuck::cast_slice(&view);
            [hits[0], hits[1]]
        };
        self.readback.unmap();
        hits
    }
}

fn update_test_parameters() -> TestParameters {
    TestParameters::default()
        .test_features_limits()
        .limits(acceleration_structure_limits())
        .features(Features::EXPERIMENTAL_RAY_QUERY)
}

#[gpu_test]
static BLAS_UPDATE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(blas_update);

/// A BLAS whose triangle moves is updated, and traces at the new position only.
fn blas_update(ctx: TestingContext) {
    let scene = UpdateScene::new(
        &ctx,
        AccelerationStructureUpdateMode::PreferUpdate,
        AccelerationStructureUpdateMode::Build,
    );

    scene.write_triangles(&ctx, &[triangle(0.0)]);
    ctx.queue.submit([scene.encode_build(&ctx, 3)]);
    assert_eq!(scene.trace(&ctx), HIT_A);

    // Same inputs, moved vertices: an update.
    scene.write_triangles(&ctx, &[triangle(10.0)]);
    ctx.queue.submit([scene.encode_build(&ctx, 3)]);
    assert_eq!(scene.trace(&ctx), HIT_B);

    // That build was an update: one encoded now and submitted after a build with other inputs
    // has lost its source, and submission rejects it.
    let update = scene.encode_build(&ctx, 3);
    ctx.queue.submit([scene.encode_build(&ctx, MAX_VERTICES)]);
    fail(
        &ctx.device,
        || ctx.queue.submit([update]),
        Some("was encoded as an update of its last build"),
    );
}

#[gpu_test]
static TLAS_UPDATE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(tlas_update);

/// A TLAS whose instance moves, with the same instance count, is updated, and traces the
/// instance at its new position only.
fn tlas_update(ctx: TestingContext) {
    let mut scene = UpdateScene::new(
        &ctx,
        AccelerationStructureUpdateMode::Build,
        AccelerationStructureUpdateMode::PreferUpdate,
    );

    scene.write_triangles(&ctx, &[triangle(0.0)]);
    ctx.queue.submit([scene.encode_build(&ctx, 3)]);
    assert_eq!(scene.trace(&ctx), HIT_A);

    // Same instance count, moved instance: an update.
    scene.tlas[0].as_mut().unwrap().transform = translation(10.0);
    ctx.queue.submit([scene.encode_tlas_build(&ctx)]);
    assert_eq!(scene.trace(&ctx), HIT_B);

    // That build was an update: one encoded now and submitted after a build with another
    // instance count has lost its source, and submission rejects it.
    let update = scene.encode_tlas_build(&ctx);
    scene.tlas[1] = Some(TlasInstance::new(&scene.blas, translation(20.0), 0, 0xFF));
    ctx.queue.submit([scene.encode_tlas_build(&ctx)]);
    fail(
        &ctx.device,
        || ctx.queue.submit([update]),
        Some("was encoded as an update of its last build"),
    );

    // A changed instance count is a full build, which is correct.
    scene.tlas[1] = None;
    ctx.queue.submit([scene.encode_tlas_build(&ctx)]);
    assert_eq!(scene.trace(&ctx), HIT_B);
}

#[gpu_test]
static BLAS_INCOMPATIBLE_BUILD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(blas_incompatible_build);

/// A build that changes the vertex count of a BLAS preferring updates is a full build.
fn blas_incompatible_build(ctx: TestingContext) {
    let scene = UpdateScene::new(
        &ctx,
        AccelerationStructureUpdateMode::PreferUpdate,
        AccelerationStructureUpdateMode::Build,
    );

    scene.write_triangles(&ctx, &[triangle(0.0)]);
    ctx.queue.submit([scene.encode_build(&ctx, 3)]);
    assert_eq!(scene.trace(&ctx), HIT_A);

    // Six vertices instead of three: a full build. It stays valid when submitted after a build
    // with yet other inputs (nine vertices), which an update would not.
    scene.write_triangles(&ctx, &[triangle(10.0), triangle(30.0), triangle(50.0)]);
    let build = scene.encode_build(&ctx, 6);
    ctx.queue.submit([scene.encode_build(&ctx, MAX_VERTICES)]);
    ctx.queue.submit([build]);
    assert_eq!(scene.trace(&ctx), HIT_B);
}

#[gpu_test]
static BLAS_REQUESTED_REBUILD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(blas_requested_rebuild);

/// `Blas::request_rebuild` makes the next build a full one, and only the next.
fn blas_requested_rebuild(ctx: TestingContext) {
    let scene = UpdateScene::new(
        &ctx,
        AccelerationStructureUpdateMode::PreferUpdate,
        AccelerationStructureUpdateMode::Build,
    );

    scene.write_triangles(&ctx, &[triangle(0.0)]);
    ctx.queue.submit([scene.encode_build(&ctx, 3)]);
    assert_eq!(scene.trace(&ctx), HIT_A);

    // A requested rebuild with the same inputs is a full build, which stays valid when submitted
    // after a build with other inputs.
    scene.write_triangles(&ctx, &[triangle(10.0)]);
    scene.blas.request_rebuild();
    let rebuild = scene.encode_build(&ctx, 3);
    ctx.queue.submit([scene.encode_build(&ctx, MAX_VERTICES)]);
    ctx.queue.submit([rebuild]);
    assert_eq!(scene.trace(&ctx), HIT_B);

    // The request was taken: the next build with the same inputs is an update again.
    let update = scene.encode_build(&ctx, 3);
    ctx.queue.submit([scene.encode_build(&ctx, MAX_VERTICES)]);
    fail(
        &ctx.device,
        || ctx.queue.submit([update]),
        Some("was encoded as an update of its last build"),
    );
}

#[gpu_test]
static PREFER_UPDATE_WITHOUT_ALLOW_UPDATE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters().enable_noop())
    .run_sync(prefer_update_without_allow_update);

/// `PreferUpdate` requires `ALLOW_UPDATE`.
fn prefer_update_without_allow_update(ctx: TestingContext) {
    fail(
        &ctx.device,
        || {
            ctx.device.create_blas(
                &CreateBlasDescriptor {
                    label: None,
                    flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: AccelerationStructureUpdateMode::PreferUpdate,
                },
                BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![size_descriptor(3)],
                },
            )
        },
        Some("requires the flag"),
    );
    fail(
        &ctx.device,
        || {
            ctx.device.create_tlas(&CreateTlasDescriptor {
                label: None,
                max_instances: 1,
                flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: AccelerationStructureUpdateMode::PreferUpdate,
            })
        },
        Some("requires the flag"),
    );
}

#[gpu_test]
static COMPACT_WHILE_SUBMITTING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(compact_while_submitting);

/// `Queue::compact_blas` while another thread submits must not deadlock: both take the device's
/// command indices and the queue's pending writes, so they must take them in the same order.
fn compact_while_submitting(ctx: TestingContext) {
    const COMPACTIONS: usize = 500;

    let scene = UpdateScene::new(
        &ctx,
        AccelerationStructureUpdateMode::Build,
        AccelerationStructureUpdateMode::Build,
    );
    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: Some("compacted BLAS"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE
                | AccelerationStructureFlags::ALLOW_COMPACTION,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![size_descriptor(3)],
        },
    );
    scene.write_triangles(&ctx, &[triangle(0.0)]);
    let size = size_descriptor(3);
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.build_acceleration_structures(
        [&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                size: &size,
                vertex_buffer: &scene.vertices,
                first_vertex: 0,
                vertex_stride: mem::size_of::<[f32; 3]>() as BufferAddress,
                index_buffer: None,
                first_index: None,
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }],
        [],
    );
    ctx.queue.submit([encoder.finish()]);
    blas.prepare_compaction_async(|result| result.unwrap());
    ctx.device.poll(PollType::wait_indefinitely()).unwrap();
    assert!(blas.ready_for_compaction());

    // The BLAS stays ready after compaction, so it is compacted again and again while another
    // thread submits. A deadlock leaves both threads blocked, so the main thread only waits
    // with a timeout.
    let stop = Arc::new(AtomicBool::new(false));
    let submitter = {
        let device = ctx.device.clone();
        let queue = ctx.queue.clone();
        let stop = stop.clone();
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                queue.submit([]);
                device.poll(PollType::Poll).unwrap();
            }
        })
    };
    let (done, finished) = mpsc::channel();
    let compactor = {
        let queue = ctx.queue.clone();
        std::thread::spawn(move || {
            for _ in 0..COMPACTIONS {
                drop(queue.compact_blas(&blas));
            }
            done.send(()).unwrap();
        })
    };

    let result = finished.recv_timeout(Duration::from_secs(60));
    stop.store(true, Ordering::Relaxed);
    if result.is_err() {
        // Dropping the device would block on the deadlocked locks too.
        mem::forget(ctx);
        panic!("compact_blas and queue.submit deadlocked (no progress in 60 s)");
    }
    compactor.join().unwrap();
    submitter.join().unwrap();
    ctx.device.poll(PollType::wait_indefinitely()).unwrap();
}
