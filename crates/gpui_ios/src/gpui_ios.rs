//! Minimal iOS backend for GPUI (spike/gpui-ios-poc).
//!
//! A `gpui_web`-shaped backend for iOS: portable `gpui_wgpu` renderer on a
//! Metal `wgpu` device, a GCD dispatcher, and cosmic-text — host-driven under
//! the Swift app's `UIApplicationMain`. This is a GO/NO-GO spike (issue #1325
//! de-risk), NOT the production backend; it implements only what the two PoC
//! questions need: (A) the run-loop under a real app lifecycle, and (B) handing
//! chart-engine GPUI's Metal device via [`PlatformWindow::shared_wgpu`].
//!
//! Written from scratch against the fork's Apache-2.0 `gpui_web`/`gpui_wgpu`,
//! learning the mechanics from `references/gpui-mobile` (we elect Apache-2.0
//! from its disjunctive license); no source is copied.
//!
//! # Consumer contract
//! The Swift host calls a single boot entry (via the app crate's `extern "C"`
//! wrapper) after registering a root view with [`set_root_view`], then drives
//! [`request_frame`], [`touch`], [`resize`], and [`set_active`] each
//! frame/event. See `ffi.rs`.

// Rust guideline compliant 2026-02-21

// The GPUI/UIKit backend proper is iOS-only: its `gpui_wgpu` Metal surface, GCD
// dispatcher, and `objc`-shaped FFI have no meaning off-device, so those modules
// are `cfg(target_os = "ios")`. The `text_seam` module — the portable half of
// the text-input platform seam (handler storage + keystroke forwarding, no
// UIKit) — compiles on every target so its headless `#[gpui::test]` runs on the
// host.

// Portable: the text-input seam's handler storage + forwarding logic.
mod text_seam;

#[cfg(target_os = "ios")]
mod dispatcher;
#[cfg(target_os = "ios")]
mod display;
#[cfg(target_os = "ios")]
mod ffi;
#[cfg(target_os = "ios")]
mod momentum;
#[cfg(target_os = "ios")]
mod platform;
#[cfg(target_os = "ios")]
mod touch;
#[cfg(target_os = "ios")]
mod window;

#[cfg(target_os = "ios")]
pub use ffi::{
    boot, delete_backward, insert_text, keyboard_caret, keyboard_wanted, request_frame, resize,
    safe_area_insets, set_active, set_root_view, set_safe_area_insets, touch,
};
#[cfg(target_os = "ios")]
pub use platform::IosPlatform;
