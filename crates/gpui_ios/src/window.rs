//! [`PlatformWindow`] for iOS: a Swift-owned `CAMetalLayer` view rendered by
//! `gpui_wgpu` on a Metal `wgpu` device.
//!
//! The Swift host (`UIApplicationMain`) owns the `UIView` (whose `layerClass`
//! is `CAMetalLayer`), the `CADisplayLink`, and the run loop. It hands this
//! backend the raw `CAMetalLayer` pointer + physical size + scale. We build a
//! Metal `wgpu::Instance` and a surface from the layer
//! (`WgpuRenderer::new_from_metal_layer`, the Simulator-proven
//! `CoreAnimationLayer` path) + a [`WgpuRenderer`] — the `gpui_web` window
//! contract, swapping the canvas for the layer and RAF for the host
//! `CADisplayLink`.
//!
//! The frame pump, touch, resize, and lifecycle are driven from the FFI layer
//! (see `ffi.rs`), which reaches the shared [`IosWindowState`] registered at
//! open-window time.

// Rust guideline compliant 2026-02-21

use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::Result;
use gpui::{
    AnyWindowHandle, Bounds, Capslock, DevicePixels, DispatchEventResult, GpuSpecs, Modifiers,
    Pixels, PlatformAtlas, PlatformDisplay, PlatformInput, PlatformInputHandler, PlatformWindow,
    Point, PromptButton, PromptLevel, RequestFrameOptions, Scene, Size, WindowAppearance,
    WindowBackgroundAppearance, WindowBounds, WindowControlArea, WindowParams, px,
};
use gpui_wgpu::{GpuContext, WgpuRenderer, WgpuSurfaceConfig};
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, UiKitDisplayHandle, WindowHandle,
};

/// GPUI-registered window callbacks (only the ones the PoC drives are stored).
#[derive(Default)]
struct Callbacks {
    request_frame: Option<Box<dyn FnMut(RequestFrameOptions)>>,
    input: Option<Box<dyn FnMut(PlatformInput) -> DispatchEventResult>>,
    active_status_change: Option<Box<dyn FnMut(bool)>>,
    resize: Option<Box<dyn FnMut(Size<Pixels>, f32)>>,
    appearance_changed: Option<Box<dyn FnMut()>>,
}

/// Shared, main-thread-only window state reached by both GPUI (via
/// [`IosWindow`]) and the FFI frame/touch/resize entry points.
pub(crate) struct IosWindowState {
    renderer: RefCell<WgpuRenderer>,
    callbacks: RefCell<Callbacks>,
    bounds: Cell<Bounds<Pixels>>,
    scale_factor: Cell<f32>,
    mouse_position: Cell<Point<Pixels>>,
    appearance: Cell<WindowAppearance>,
}

impl IosWindowState {
    /// Fire the GPUI frame callback (take-invoke-restore to avoid a live borrow
    /// while GPUI re-enters `draw`). Called once per `CADisplayLink` tick.
    pub(crate) fn request_frame(&self) {
        let cb = self.callbacks.borrow_mut().request_frame.take();
        if let Some(mut cb) = cb {
            cb(RequestFrameOptions {
                require_presentation: true,
                // The PoC drives a continuously-redrawing embedded chart, so
                // force a redraw each tick rather than only on invalidation.
                force_render: true,
            });
            self.callbacks.borrow_mut().request_frame = Some(cb);
        }
    }

    /// Deliver a synthesized [`PlatformInput`] to GPUI's input sink.
    pub(crate) fn dispatch_input(&self, input: PlatformInput) {
        if let PlatformInput::MouseMove(ref ev) = input {
            self.mouse_position.set(ev.position);
        }
        if let PlatformInput::MouseDown(ref ev) = input {
            self.mouse_position.set(ev.position);
        }
        let cb = self.callbacks.borrow_mut().input.take();
        if let Some(mut cb) = cb {
            cb(input);
            self.callbacks.borrow_mut().input = Some(cb);
        }
    }

    /// Reconfigure the drawable + notify GPUI on rotation / layout change.
    pub(crate) fn handle_resize(&self, physical_w: u32, physical_h: u32, scale: f32) {
        self.scale_factor.set(scale);
        let logical = Size {
            width: px(physical_w as f32 / scale),
            height: px(physical_h as f32 / scale),
        };
        self.bounds.set(Bounds {
            origin: Point::default(),
            size: logical,
        });
        self.renderer.borrow_mut().update_drawable_size(Size {
            width: DevicePixels(physical_w as i32),
            height: DevicePixels(physical_h as i32),
        });
        let cb = self.callbacks.borrow_mut().resize.take();
        if let Some(mut cb) = cb {
            cb(logical, scale);
            self.callbacks.borrow_mut().resize = Some(cb);
        }
    }

    /// Notify GPUI the app moved to fore/background (pause pump in bg).
    pub(crate) fn notify_active(&self, active: bool) {
        let cb = self.callbacks.borrow_mut().active_status_change.take();
        if let Some(mut cb) = cb {
            cb(active);
            self.callbacks.borrow_mut().active_status_change = Some(cb);
        }
    }
}

/// iOS platform window. Holds the shared state and its display.
pub(crate) struct IosWindow {
    state: Rc<IosWindowState>,
    display: Rc<dyn PlatformDisplay>,
}

impl IosWindow {
    /// Build the window from the Swift-owned `CAMetalLayer` pointer.
    pub(crate) fn new(
        _handle: AnyWindowHandle,
        params: WindowParams,
        display: Rc<dyn PlatformDisplay>,
        metal_layer: *mut c_void,
        physical_w: u32,
        physical_h: u32,
        scale: f32,
    ) -> Result<Self> {
        let config = WgpuSurfaceConfig {
            size: Size {
                width: DevicePixels(physical_w as i32),
                height: DevicePixels(physical_h as i32),
            },
            transparent: false,
            // Mailbox avoids blocking `get_current_texture` across iOS
            // lifecycle transitions; falls back to Fifo if unsupported.
            preferred_present_mode: Some(wgpu::PresentMode::Mailbox),
        };

        // Surface via the CAMetalLayer directly — the Simulator-proven path
        // (`SurfaceTargetUnsafe::CoreAnimationLayer`). The renderer builds its
        // own Metal instance + context on an empty `GpuContext`.
        let gpu_context: GpuContext = Rc::new(RefCell::new(None));
        // SAFETY: the layer is owned by the Swift host for the process lifetime.
        let renderer =
            unsafe { WgpuRenderer::new_from_metal_layer(gpu_context, metal_layer, config, None)? };

        // The CAMetalLayer defines the real drawable; logical bounds are the
        // physical size divided by scale. `params.bounds` is advisory on a
        // full-screen iOS surface and intentionally ignored.
        let _ = params;
        let logical = Size {
            width: px(physical_w as f32 / scale),
            height: px(physical_h as f32 / scale),
        };
        let bounds = Bounds {
            origin: Point::default(),
            size: logical,
        };

        let state = Rc::new(IosWindowState {
            renderer: RefCell::new(renderer),
            callbacks: RefCell::new(Callbacks::default()),
            bounds: Cell::new(bounds),
            scale_factor: Cell::new(scale),
            mouse_position: Cell::new(Point::default()),
            appearance: Cell::new(WindowAppearance::Dark),
        });

        Ok(Self { state, display })
    }

    /// The shared state, registered with the FFI layer so the host run loop can
    /// drive frames/touches/resize.
    pub(crate) fn state(&self) -> Rc<IosWindowState> {
        self.state.clone()
    }
}

impl HasWindowHandle for IosWindow {
    fn window_handle(&self) -> std::result::Result<WindowHandle<'_>, HandleError> {
        Err(HandleError::NotSupported)
    }
}

impl HasDisplayHandle for IosWindow {
    fn display_handle(&self) -> std::result::Result<DisplayHandle<'_>, HandleError> {
        let handle = UiKitDisplayHandle::new();
        // SAFETY: carries no borrowed data.
        Ok(unsafe { DisplayHandle::borrow_raw(handle.into()) })
    }
}

impl PlatformWindow for IosWindow {
    fn bounds(&self) -> Bounds<Pixels> {
        self.state.bounds.get()
    }

    fn is_maximized(&self) -> bool {
        true
    }

    fn window_bounds(&self) -> WindowBounds {
        WindowBounds::Fullscreen(self.state.bounds.get())
    }

    fn content_size(&self) -> Size<Pixels> {
        self.state.bounds.get().size
    }

    fn resize(&mut self, _size: Size<Pixels>) {}

    fn scale_factor(&self) -> f32 {
        self.state.scale_factor.get()
    }

    fn appearance(&self) -> WindowAppearance {
        self.state.appearance.get()
    }

    fn display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        Some(self.display.clone())
    }

    fn mouse_position(&self) -> Point<Pixels> {
        self.state.mouse_position.get()
    }

    fn modifiers(&self) -> Modifiers {
        Modifiers::default()
    }

    fn capslock(&self) -> Capslock {
        Capslock::default()
    }

    fn set_input_handler(&mut self, _input_handler: PlatformInputHandler) {}

    fn take_input_handler(&mut self) -> Option<PlatformInputHandler> {
        None
    }

    fn prompt(
        &self,
        _level: PromptLevel,
        _msg: &str,
        _detail: Option<&str>,
        _answers: &[PromptButton],
    ) -> Option<futures::channel::oneshot::Receiver<usize>> {
        None
    }

    fn activate(&self) {}

    fn is_active(&self) -> bool {
        true
    }

    fn is_hovered(&self) -> bool {
        false
    }

    fn background_appearance(&self) -> WindowBackgroundAppearance {
        WindowBackgroundAppearance::Opaque
    }

    fn set_title(&mut self, _title: &str) {}

    fn set_background_appearance(&self, _background: WindowBackgroundAppearance) {}

    fn minimize(&self) {}

    fn zoom(&self) {}

    fn toggle_fullscreen(&self) {}

    fn is_fullscreen(&self) -> bool {
        true
    }

    fn on_request_frame(&self, callback: Box<dyn FnMut(RequestFrameOptions)>) {
        self.state.callbacks.borrow_mut().request_frame = Some(callback);
    }

    fn on_input(&self, callback: Box<dyn FnMut(PlatformInput) -> DispatchEventResult>) {
        self.state.callbacks.borrow_mut().input = Some(callback);
    }

    fn on_active_status_change(&self, callback: Box<dyn FnMut(bool)>) {
        self.state.callbacks.borrow_mut().active_status_change = Some(callback);
    }

    fn on_hover_status_change(&self, _callback: Box<dyn FnMut(bool)>) {}

    fn on_resize(&self, callback: Box<dyn FnMut(Size<Pixels>, f32)>) {
        self.state.callbacks.borrow_mut().resize = Some(callback);
    }

    fn on_moved(&self, _callback: Box<dyn FnMut()>) {}

    fn on_should_close(&self, _callback: Box<dyn FnMut() -> bool>) {}

    fn on_close(&self, _callback: Box<dyn FnOnce()>) {}

    fn on_hit_test_window_control(&self, _callback: Box<dyn FnMut() -> Option<WindowControlArea>>) {
    }

    fn on_appearance_changed(&self, callback: Box<dyn FnMut()>) {
        self.state.callbacks.borrow_mut().appearance_changed = Some(callback);
    }

    fn draw(&self, scene: &Scene) {
        self.state.renderer.borrow_mut().draw(scene);
    }

    fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        self.state.renderer.borrow().sprite_atlas().clone()
    }

    // Lux fork (additive): hand chart-engine GPUI's Metal device/queue/adapter
    // for the zero-copy embed (the whole point of question B). iOS is
    // `not(target_os = "macos")`, so this override is available exactly as on
    // web — the seam chart-engine's `SharedGpu::from_shared_wgpu` consumes.
    fn shared_wgpu(&self) -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>, wgpu::Adapter)> {
        Some(self.state.renderer.borrow().shared_gpu())
    }

    fn is_subpixel_rendering_supported(&self) -> bool {
        self.state.renderer.borrow().supports_dual_source_blending()
    }

    fn gpu_specs(&self) -> Option<GpuSpecs> {
        Some(self.state.renderer.borrow().gpu_specs())
    }

    fn update_ime_position(&self, _bounds: Bounds<Pixels>) {}
}
