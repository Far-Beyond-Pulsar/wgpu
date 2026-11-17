use alloc::vec::Vec;

#[cfg(any(feature = "serde", test))]
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::{Features, VertexFormat};

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for all size defining attributes of a single triangle geometry inside a bottom level acceleration structure.
pub struct BlasTriangleGeometrySizeDescriptor {
    /// Format of a vertex position, must be [`VertexFormat::Float32x3`]
    /// with just [`Features::EXPERIMENTAL_RAY_QUERY`]
    /// but [`Features::EXTENDED_ACCELERATION_STRUCTURE_VERTEX_FORMATS`] adds more.
    pub vertex_format: crate::VertexFormat,
    /// Number of vertices.
    pub vertex_count: u32,
    /// Format of an index. Only needed if an index buffer is used.
    /// If `index_format` is provided `index_count` is required.
    pub index_format: Option<crate::IndexFormat>,
    /// Number of indices. Only needed if an index buffer is used.
    /// If `index_count` is provided `index_format` is required.
    pub index_count: Option<u32>,
    /// Flags for the geometry.
    pub flags: AccelerationStructureGeometryFlags,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for size-defining attributes of one AABB geometry group in a bottom level acceleration structure.
///
/// Each primitive is one axis-aligned bounding box, typically stored as two `vec3<f32>` min/max corners.
pub struct BlasAABBGeometrySizeDescriptor {
    /// Number of AABB primitives in this geometry.
    pub primitive_count: u32,
    /// Flags for the geometry.
    pub flags: AccelerationStructureGeometryFlags,
}

/// Minimum stride for AABB geometry (24 bytes: two `vec3<f32>`).
pub const AABB_GEOMETRY_MIN_STRIDE: crate::BufferAddress = 24;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for all size defining attributes of all geometries inside a bottom level acceleration structure.
pub enum BlasGeometrySizeDescriptors {
    /// Triangle geometry version.
    Triangles {
        /// Descriptor for each triangle geometry.
        descriptors: Vec<BlasTriangleGeometrySizeDescriptor>,
    },
    /// AABB geometry version.
    AABBs {
        /// Descriptor for each AABB geometry.
        descriptors: Vec<BlasAABBGeometrySizeDescriptor>,
    },
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Update mode for acceleration structure builds.
pub enum AccelerationStructureUpdateMode {
    /// Always perform a full build.
    Build,
    /// Update (refit) the structure in place when a build allows it, else perform a full build.
    ///
    /// An update keeps the tree of the last full build and only moves it to the new input
    /// data. It is much cheaper than a build, but trace performance drops as the inputs drift
    /// from those of the last full build, so it suits deforming geometry (skinning, cloth) and
    /// moving instances, with a full build every few frames (`Blas::request_rebuild`,
    /// `Tlas::request_rebuild` in `wgpu`).
    ///
    /// Requires [`AccelerationStructureFlags::ALLOW_UPDATE`].
    ///
    /// A build is an update when the structure was last built by `wgpu` (not by `as_hal` code
    /// marked with `mark_acceleration_structures_built`), no rebuild was requested, and:
    /// - for a BLAS, the build keeps, per geometry, the size descriptor (vertex format and
    ///   count, index format and count, AABB count, flags), the index buffer and first index,
    ///   and whether a transform buffer is used, as well as the number and kind of geometries.
    ///   Vertex data (buffer, first vertex, stride, positions), transform contents and AABB data
    ///   may change.
    /// - for a TLAS, the build keeps the number of instances. Transforms, masks, custom data and
    ///   the BLAS each instance refers to may change.
    ///
    /// The caller must also keep the index values of an indexed BLAS, which `wgpu` cannot check,
    /// and must not make a triangle active or inactive (a vertex with a NaN position makes it
    /// inactive). Request a rebuild when either changes.
    PreferUpdate,
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for creating a bottom level acceleration structure.
pub struct CreateBlasDescriptor<L> {
    /// Label for the bottom level acceleration structure.
    pub label: L,
    /// Flags for the bottom level acceleration structure.
    pub flags: AccelerationStructureFlags,
    /// Update mode for the bottom level acceleration structure.
    pub update_mode: AccelerationStructureUpdateMode,
}

impl<L> CreateBlasDescriptor<L> {
    /// Takes a closure and maps the label of the blas descriptor into another.
    pub fn map_label<'a, K>(&'a self, fun: impl FnOnce(&'a L) -> K) -> CreateBlasDescriptor<K> {
        CreateBlasDescriptor {
            label: fun(&self.label),
            flags: self.flags,
            update_mode: self.update_mode,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for creating a top level acceleration structure.
pub struct CreateTlasDescriptor<L> {
    /// Label for the top level acceleration structure.
    pub label: L,
    /// Number of instances that can be stored in the acceleration structure.
    pub max_instances: u32,
    /// Flags for the bottom level acceleration structure.
    pub flags: AccelerationStructureFlags,
    /// Update mode for the bottom level acceleration structure.
    pub update_mode: AccelerationStructureUpdateMode,
}

impl<L> CreateTlasDescriptor<L> {
    /// Takes a closure and maps the label of the blas descriptor into another.
    pub fn map_label<'a, K>(&'a self, fun: impl FnOnce(&'a L) -> K) -> CreateTlasDescriptor<K> {
        CreateTlasDescriptor {
            label: fun(&self.label),
            flags: self.flags,
            update_mode: self.update_mode,
            max_instances: self.max_instances,
        }
    }
}

bitflags::bitflags!(
    /// Flags for acceleration structures
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct AccelerationStructureFlags: u8 {
        /// Allow the acceleration structure to be updated (refit) instead of rebuilt. Required
        /// by [`AccelerationStructureUpdateMode::PreferUpdate`]; it may make builds slower and
        /// the structure larger, so set it only on structures that are updated.
        const ALLOW_UPDATE = 1 << 0;
        /// Allow the acceleration structure to be compacted in a copy operation
        /// (`Blas::prepare_for_compaction`, `CommandEncoder::compact_blas`).
        const ALLOW_COMPACTION = 1 << 1;
        /// Optimize for fast ray tracing performance, recommended if the geometry is unlikely
        /// to change (e.g. in a game: non-interactive scene geometry)
        const PREFER_FAST_TRACE = 1 << 2;
        /// Optimize for fast build time, recommended if geometry is likely to change frequently
        /// (e.g. in a game: player model).
        const PREFER_FAST_BUILD = 1 << 3;
        /// Optimize for low memory footprint (both while building and in the output BLAS).
        const LOW_MEMORY = 1 << 4;
        /// Use `BlasTriangleGeometry::transform_buffer` when building a BLAS (only allowed in
        /// BLAS creation)
        const USE_TRANSFORM = 1 << 5;
        /// Allow retrieval of the vertices of the triangle hit by a ray.
        const ALLOW_RAY_HIT_VERTEX_RETURN = 1 << 6;
    }
);

bitflags::bitflags!(
    /// Flags for acceleration structure geometries
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct AccelerationStructureGeometryFlags: u8 {
        /// Is OPAQUE (is there no alpha test) recommended as currently in naga there is no
        /// candidate intersections yet so currently BLASes without this flag will not have hits.
        /// Not enabling this makes the BLAS unable to be interacted with in WGSL.
        const OPAQUE = 1 << 0;
        /// NO_DUPLICATE_ANY_HIT_INVOCATION, not useful unless using hal with wgpu, ray-tracing
        /// pipelines are not supported in wgpu so any-hit shaders do not exist. For when any-hit
        /// shaders are implemented (or experienced users who combine this with an underlying library:
        /// for any primitive (triangle or AABB) multiple any-hit shaders sometimes may be invoked
        /// (especially in AABBs like a sphere), if this flag in present only one hit on a primitive may
        /// invoke an any-hit shader.
        const NO_DUPLICATE_ANY_HIT_INVOCATION = 1 << 1;
    }
);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// What a copy between acceleration structures should do
pub enum AccelerationStructureCopy {
    /// Directly duplicate an acceleration structure to another
    Clone,
    /// Duplicate and compact an acceleration structure
    Compact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// What type the data of an acceleration structure is
pub enum AccelerationStructureType {
    /// The types of the acceleration structure are triangles
    Triangles,
    /// The types of the acceleration structure are axis aligned bounding boxes
    AABBs,
    /// The types of the acceleration structure are instances
    Instances,
}

/// Alignment requirement for transform buffers used in acceleration structure builds
pub const TRANSFORM_BUFFER_ALIGNMENT: crate::BufferAddress = 16;

/// Alignment requirement for instance buffers used in acceleration structure builds (`build_acceleration_structures_unsafe_tlas`)
pub const INSTANCE_BUFFER_ALIGNMENT: crate::BufferAddress = 16;
