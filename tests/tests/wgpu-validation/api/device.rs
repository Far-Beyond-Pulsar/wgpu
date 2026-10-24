/// Test that if another error occurs reentrantly in the [`wgpu::Device::on_uncaptured_error`]
/// handler, this does not result in a deadlock (as a previous implementation would have had).
#[cfg(not(target_family = "wasm"))] // test needs wgpu::Device: Send + Sync to achieve reentrance
#[test]
fn recursive_uncaptured_error() {
    use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
    use std::sync::Arc;

    const ERRONEOUS_TEXTURE_DESC: wgpu::TextureDescriptor = wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 10,
        },
        mip_level_count: 0,
        sample_count: 0,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };

    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let errors_seen: Arc<AtomicU32> = Arc::default();
    let handler = Arc::new({
        let errors_seen = errors_seen.clone();
        let device = device.clone();
        move |_error| {
            let previous_count = errors_seen.fetch_add(1, Relaxed);
            if previous_count == 0 {
                // Trigger another error recursively
                _ = device.create_texture(&ERRONEOUS_TEXTURE_DESC);
            }
        }
    });

    // Trigger one error which will call the handler
    device.on_uncaptured_error(handler);
    _ = device.create_texture(&ERRONEOUS_TEXTURE_DESC);

    assert_eq!(errors_seen.load(Relaxed), 2);
}

/// HELIO PATCH: handles of separate instances are different handles, though
/// their ids (unique only within an instance) match.
#[test]
fn handles_of_separate_instances_differ() {
    use std::collections::HashSet;

    // Each `noop` device is the first device of its own instance.
    let (first, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let (second, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    assert_ne!(first, second);
    assert_eq!(first, first.clone());
    let devices: HashSet<_> = [first.clone(), second.clone(), first.clone()].into();
    assert_eq!(devices.len(), 2);

    let buffer = |device: &wgpu::Device| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    assert_ne!(buffer(&first), buffer(&second));
}
