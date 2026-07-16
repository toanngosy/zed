//! C-ABI seam between the Swift host and the GPUI iOS backend.
//!
//! All entry points run on the iOS main thread (the app delegate, the
//! `CADisplayLink` target, and `UIView` touch overrides all live there), so
//! shared state is kept in `thread_local!` cells — no `unsafe` `Send`/`Sync`
//! and no cross-thread races. The host drives four things: boot (once),
//! per-frame ticks, touches, and resize/lifecycle.
//!
//! The consumer app (e.g. the PoC crate) registers its root view with
//! [`set_root_view`] and calls [`boot`] from its own `extern "C"` entry; Swift
//! then calls the `gpui_ios_*` symbols below each frame/touch.
//!
//! # Touch ABI
//!
//! [`touch`] carries `(pointer_id, phase, x, y, scale)`. `phase` is **our own**
//! `0/1/2/3` convention (began / moved / ended / cancelled), NOT raw
//! `UITouchPhase` — the Swift shim MUST translate. `pointer_id` is a stable
//! per-finger identity so two fingers can be tracked for pinch. The full
//! contract lives at the aggregator ([`crate::touch`]).
//!
//! # Abort
//!
//! These functions are reached from the consumer's `extern "C"` trampolines. A
//! panic that unwinds into a trampoline crosses the C ABI and **aborts** the
//! process — there is intentionally no `catch_unwind` here, matching
//! `gpui_macos` / `gpui_web` (neither installs one at its platform boundary).
//! Keep these paths panic-free.

// Rust guideline compliant 2026-02-21

use std::cell::RefCell;
use std::ffi::c_void;
use std::rc::Rc;

use gpui::{App, AppContext, Application, Edges, Pixels, Point, px};

use crate::platform::IosPlatform;
use crate::window::IosWindowState;

/// Physical drawable size + scale + `CAMetalLayer` pointer handed over by Swift.
#[derive(Clone, Copy)]
pub(crate) struct BootParams {
    pub(crate) metal_layer: *mut c_void,
    pub(crate) physical_w: u32,
    pub(crate) physical_h: u32,
    pub(crate) scale: f32,
}

thread_local! {
    /// Boot parameters, read by `IosPlatform::open_window`.
    static BOOT_PARAMS: RefCell<Option<BootParams>> = const { RefCell::new(None) };
    /// The GPUI finish-launching closure, parked until the host is ready.
    static FINISH_LAUNCHING: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
    /// The one window's shared state, driven by the frame/touch/resize FFI.
    static WINDOW_STATE: RefCell<Option<Rc<IosWindowState>>> = const { RefCell::new(None) };
    /// The consumer's root-view builder, invoked inside the run closure.
    static APP_CALLBACK: RefCell<Option<Box<dyn FnOnce(&mut App)>>> = const { RefCell::new(None) };
    /// The latest UIKit safe-area insets (logical points), pushed by the host on
    /// `viewDidLayoutSubviews` / `safeAreaInsetsDidChange`. Zero until the first
    /// report; read each frame by shell code through [`safe_area_insets`]. Not a
    /// `const` initializer because `Edges::default()` is not `const`.
    static SAFE_AREA_INSETS: RefCell<Edges<Pixels>> = RefCell::new(Edges::default());
}

/// Register the consumer's root-view builder. Call before [`boot`].
pub fn set_root_view(builder: Box<dyn FnOnce(&mut App)>) {
    APP_CALLBACK.with(|c| *c.borrow_mut() = Some(builder));
}

/// Store the boot parameters (called by [`boot`] before running the app).
pub(crate) fn set_boot_params(params: BootParams) {
    BOOT_PARAMS.with(|c| *c.borrow_mut() = Some(params));
}

/// Read the boot parameters (called by `IosPlatform::open_window`).
pub(crate) fn boot_params() -> Option<BootParams> {
    BOOT_PARAMS.with(|c| *c.borrow())
}

/// Park the GPUI finish-launching closure (called by `IosPlatform::run`).
pub(crate) fn store_finish_launching(cb: Box<dyn FnOnce()>) {
    FINISH_LAUNCHING.with(|c| *c.borrow_mut() = Some(cb));
}

/// Register the window state so the FFI frame/touch/resize path can reach it.
pub(crate) fn register_window(state: Rc<IosWindowState>) {
    WINDOW_STATE.with(|c| *c.borrow_mut() = Some(state));
}

/// Boot the GPUI iOS app under the host's `UIApplicationMain`.
///
/// Stashes the drawable parameters, runs the GPUI application (which — on this
/// host-driven platform — parks the finish-launching closure and returns), then
/// fires that closure so the window opens. The app is then kept alive by the
/// window callbacks GPUI installed during `open_window` (a deliberate retain
/// cycle for a process-lifetime app, mirroring the reference backend).
///
/// # Safety
/// `metal_layer` must be a valid `CAMetalLayer*` alive for the process
/// lifetime. Must be called exactly once, on the main thread.
pub unsafe fn boot(metal_layer: *mut c_void, physical_w: u32, physical_h: u32, scale: f32) {
    set_boot_params(BootParams {
        metal_layer,
        physical_w,
        physical_h,
        scale,
    });

    let platform = Rc::new(IosPlatform::new());
    // `run_retained`: host-driven `run` returns immediately, so leak one strong
    // ref to keep the App + window alive for the process (else the frame pump
    // upgrades a released Weak and logs "app was released" every tick).
    Application::with_platform(platform).run_retained(|cx: &mut App| {
        let builder = APP_CALLBACK.with(|c| c.borrow_mut().take());
        if let Some(builder) = builder {
            builder(cx);
        } else {
            log::warn!("gpui_ios: no root view registered; opening empty window");
            cx.open_window(gpui::WindowOptions::default(), |_, cx| {
                cx.new(|_| gpui::Empty)
            })
            .expect("failed to open default window");
        }
        cx.activate(true);
    });

    // Host-driven: `Platform::run` parked the closure. Fire it now so the
    // window actually opens (in a fuller integration the app delegate would).
    let finish = FINISH_LAUNCHING.with(|c| c.borrow_mut().take());
    if let Some(finish) = finish {
        finish();
        log::info!("gpui_ios: finish-launching fired; window open");
    }
}

/// CADisplayLink tick: render one frame.
///
/// Pumps any in-flight momentum fling first (so a synthetic drag-continuation
/// move is applied before this frame draws), then renders.
///
/// Exposed as a plain Rust fn (not `extern "C"`): the final `staticlib` (the
/// PoC crate) provides the `#[no_mangle]` wrapper, guaranteeing the C symbol
/// survives linking instead of being stripped from a dependency rlib.
pub fn request_frame() {
    WINDOW_STATE.with(|c| {
        if let Some(state) = c.borrow().as_ref() {
            state.pump_momentum();
            state.request_frame();
        }
    });
}

/// Touch event forwarded from Swift, in physical pixels.
///
/// `pointer_id` is a stable per-finger identity; `phase` is the
/// began/moved/ended/cancelled convention documented on [`crate::touch`]. The
/// aggregator turns the multi-touch stream into the cooked [`gpui::PlatformInput`]
/// events (one-finger mouse sequence, two-finger pinch, fling momentum).
pub fn touch(pointer_id: u64, phase: i32, x: f32, y: f32, scale: f32) {
    let scale = if scale > 0.0 { scale } else { 1.0 };
    // GPUI works in logical pixels; the host sends physical, so divide by scale.
    let position = Point {
        x: px(x / scale),
        y: px(y / scale),
    };
    WINDOW_STATE.with(|c| {
        if let Some(state) = c.borrow().as_ref() {
            state.dispatch_touch(pointer_id, phase, position);
        }
    });
}

/// Layout change (rotation / split-screen): reconfigure the drawable.
pub fn resize(physical_w: u32, physical_h: u32, scale: f32) {
    WINDOW_STATE.with(|c| {
        if let Some(state) = c.borrow().as_ref() {
            state.handle_resize(physical_w, physical_h, scale);
        }
    });
}

/// Foreground/background lifecycle.
pub fn set_active(active: bool) {
    WINDOW_STATE.with(|c| {
        if let Some(state) = c.borrow().as_ref() {
            state.notify_active(active);
        }
    });
}

/// Push the current safe-area insets (logical points) from the host.
///
/// Called on `viewDidLayoutSubviews` / `safeAreaInsetsDidChange`. UIKit reports
/// `UIView.safeAreaInsets` in points (already scale-independent), so the values
/// map directly to GPUI's logical [`Pixels`] with no scale division. Argument
/// order mirrors [`Edges`] (`top`, `right`, `bottom`, `left`).
pub fn set_safe_area_insets(top: f32, right: f32, bottom: f32, left: f32) {
    SAFE_AREA_INSETS.with(|c| {
        *c.borrow_mut() = Edges {
            top: px(top),
            right: px(right),
            bottom: px(bottom),
            left: px(left),
        };
    });
}

/// The safe-area insets the host last reported (logical points), or zero before
/// the first report.
///
/// Shell code reads this each frame to keep its chrome clear of the status bar /
/// Dynamic Island / home indicator. Main-thread only (like every FFI entry
/// here), so it is safe to call from `render`.
pub fn safe_area_insets() -> Edges<Pixels> {
    SAFE_AREA_INSETS.with(|c| c.borrow().clone())
}
