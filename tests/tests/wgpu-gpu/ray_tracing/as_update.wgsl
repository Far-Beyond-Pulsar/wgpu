enable wgpu_ray_query;

@group(0) @binding(0)
var acc_struct: acceleration_structure;

// Ray origins; every ray points along +z.
@group(0) @binding(1)
var<storage, read> origins: array<vec4<f32>>;

// 1 when the ray from the same origin hits a triangle, else 0.
@group(0) @binding(2)
var<storage, read_write> hits: array<u32>;

@workgroup_size(1)
@compute
fn trace(@builtin(global_invocation_id) id: vec3<u32>) {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 100.0, origins[id.x].xyz, vec3f(0.0, 0.0, 1.0)));
    rayQueryProceed(&rq);
    let intersection = rayQueryGetCommittedIntersection(&rq);
    hits[id.x] = select(0u, 1u, intersection.kind == RAY_QUERY_INTERSECTION_TRIANGLE);
}
