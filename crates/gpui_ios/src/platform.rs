//! [`Platform`] implementation for iOS (host-driven, view-only).
//!
//! Mirrors `gpui_web::WebPlatform`: real executors (GCD dispatcher), a portable
//! `CosmicTextSystem` with bundled fonts, a single display, and `open_window`
//! building the Metal-backed [`IosWindow`]. Everything else (menus, clipboard,
//! file dialogs, cursor, credentials) is a no-op/`None`/`Err` stub — a view-only
//! chart surface needs none of it. `run` is host-driven: it parks the
//! finish-launching closure (fired later by the FFI boot path) instead of
//! owning a run loop, exactly as the web backend does under the browser.

// Rust guideline compliant 2026-02-21

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use anyhow::Result;
use futures::channel::oneshot;
use gpui::{
    Action, AnyWindowHandle, BackgroundExecutor, ClipboardItem, CursorStyle, DummyKeyboardMapper,
    ForegroundExecutor, Keymap, Menu, MenuItem, PathPromptOptions, Platform, PlatformDisplay,
    PlatformKeyboardLayout, PlatformKeyboardMapper, PlatformTextSystem, PlatformWindow, Task,
    ThermalState, WindowAppearance, WindowParams,
};

use crate::dispatcher::IosDispatcher;
use crate::display::IosDisplay;
use crate::ffi;
use crate::window::IosWindow;

/// Same bundled fonts the web backend ships, so `CosmicTextSystem` renders
/// without any system-font enumeration (unavailable/undesired on iOS).
static BUNDLED_FONTS: &[&[u8]] = &[
    include_bytes!("../../../assets/fonts/ibm-plex-sans/IBMPlexSans-Regular.ttf"),
    include_bytes!("../../../assets/fonts/ibm-plex-sans/IBMPlexSans-Italic.ttf"),
    include_bytes!("../../../assets/fonts/ibm-plex-sans/IBMPlexSans-SemiBold.ttf"),
    include_bytes!("../../../assets/fonts/ibm-plex-sans/IBMPlexSans-SemiBoldItalic.ttf"),
    include_bytes!("../../../assets/fonts/lilex/Lilex-Regular.ttf"),
    include_bytes!("../../../assets/fonts/lilex/Lilex-Bold.ttf"),
    include_bytes!("../../../assets/fonts/lilex/Lilex-Italic.ttf"),
    include_bytes!("../../../assets/fonts/lilex/Lilex-BoldItalic.ttf"),
];

/// Trivial keyboard-layout descriptor (view-only: no real keymap needed).
struct IosKeyboardLayout;

impl PlatformKeyboardLayout for IosKeyboardLayout {
    fn id(&self) -> &str {
        "us"
    }

    fn name(&self) -> &str {
        "US"
    }
}

/// The iOS GPUI platform.
pub struct IosPlatform {
    background_executor: BackgroundExecutor,
    foreground_executor: ForegroundExecutor,
    text_system: Arc<dyn PlatformTextSystem>,
    active_display: Rc<dyn PlatformDisplay>,
    active_window: RefCell<Option<AnyWindowHandle>>,
    cursor_visible: Cell<bool>,
}

impl IosPlatform {
    /// Build the platform. Boot parameters must already be stashed (the FFI
    /// boot path stores them before constructing this).
    pub fn new() -> Self {
        let dispatcher = Arc::new(IosDispatcher::new());
        let background_executor = BackgroundExecutor::new(dispatcher.clone());
        let foreground_executor = ForegroundExecutor::new(dispatcher);

        let text_system = Arc::new(gpui_wgpu::CosmicTextSystem::new_without_system_fonts(
            "IBM Plex Sans",
        ));
        let fonts = BUNDLED_FONTS
            .iter()
            .map(|bytes| Cow::Borrowed(*bytes))
            .collect();
        if let Err(error) = text_system.add_fonts(fonts) {
            log::error!("gpui_ios: failed to load bundled fonts: {error:#}");
        }
        let text_system: Arc<dyn PlatformTextSystem> = text_system;

        let (w, h) = ffi::boot_params()
            .map(|p| (p.physical_w as f32 / p.scale, p.physical_h as f32 / p.scale))
            .unwrap_or((393.0, 852.0));
        let active_display: Rc<dyn PlatformDisplay> = Rc::new(IosDisplay::new(w, h));

        Self {
            background_executor,
            foreground_executor,
            text_system,
            active_display,
            active_window: RefCell::new(None),
            cursor_visible: Cell::new(true),
        }
    }
}

impl Default for IosPlatform {
    fn default() -> Self {
        Self::new()
    }
}

impl Platform for IosPlatform {
    fn background_executor(&self) -> BackgroundExecutor {
        self.background_executor.clone()
    }

    fn foreground_executor(&self) -> ForegroundExecutor {
        self.foreground_executor.clone()
    }

    fn text_system(&self) -> Arc<dyn PlatformTextSystem> {
        self.text_system.clone()
    }

    fn run(&self, on_finish_launching: Box<dyn 'static + FnOnce()>) {
        // Host-driven: park the closure; the FFI boot path fires it once the
        // UIKit app delegate has finished launching. UIApplicationMain owns the
        // run loop — we never block here.
        ffi::store_finish_launching(on_finish_launching);
    }

    fn quit(&self) {}

    fn restart(&self, _binary_path: Option<PathBuf>) {}

    fn activate(&self, _ignoring_other_apps: bool) {}

    fn hide(&self) {}

    fn hide_other_apps(&self) {}

    fn unhide_other_apps(&self) {}

    fn displays(&self) -> Vec<Rc<dyn PlatformDisplay>> {
        vec![self.active_display.clone()]
    }

    fn primary_display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        Some(self.active_display.clone())
    }

    fn active_window(&self) -> Option<AnyWindowHandle> {
        *self.active_window.borrow()
    }

    fn open_window(
        &self,
        handle: AnyWindowHandle,
        params: WindowParams,
    ) -> Result<Box<dyn PlatformWindow>> {
        let boot = ffi::boot_params().ok_or_else(|| {
            anyhow::anyhow!("gpui_ios: boot parameters not set before open_window")
        })?;

        let window = IosWindow::new(
            handle,
            params,
            self.active_display.clone(),
            boot.metal_layer,
            boot.physical_w,
            boot.physical_h,
            boot.scale,
        )?;

        // Register the shared state so the host frame/touch/resize FFI reaches
        // this window.
        ffi::register_window(window.state());
        *self.active_window.borrow_mut() = Some(handle);
        Ok(Box::new(window))
    }

    fn window_appearance(&self) -> WindowAppearance {
        WindowAppearance::Dark
    }

    fn open_url(&self, _url: &str) {}

    fn on_open_urls(&self, _callback: Box<dyn FnMut(Vec<String>)>) {}

    fn register_url_scheme(&self, _url: &str) -> Task<Result<()>> {
        Task::ready(Ok(()))
    }

    fn prompt_for_paths(
        &self,
        _options: PathPromptOptions,
    ) -> oneshot::Receiver<Result<Option<Vec<PathBuf>>>> {
        let (tx, rx) = oneshot::channel();
        tx.send(Err(anyhow::anyhow!(
            "file dialogs are not supported on iOS"
        )))
        .ok();
        rx
    }

    fn prompt_for_new_path(
        &self,
        _directory: &Path,
        _suggested_name: Option<&str>,
    ) -> oneshot::Receiver<Result<Option<PathBuf>>> {
        let (tx, rx) = oneshot::channel();
        tx.send(Err(anyhow::anyhow!(
            "file dialogs are not supported on iOS"
        )))
        .ok();
        rx
    }

    fn can_select_mixed_files_and_dirs(&self) -> bool {
        false
    }

    fn reveal_path(&self, _path: &Path) {}

    fn open_with_system(&self, _path: &Path) {}

    fn on_quit(&self, _callback: Box<dyn FnMut()>) {}

    fn on_reopen(&self, _callback: Box<dyn FnMut()>) {}

    fn set_menus(&self, _menus: Vec<Menu>, _keymap: &Keymap) {}

    fn set_dock_menu(&self, _menu: Vec<MenuItem>, _keymap: &Keymap) {}

    fn on_app_menu_action(&self, _callback: Box<dyn FnMut(&dyn Action)>) {}

    fn on_will_open_app_menu(&self, _callback: Box<dyn FnMut()>) {}

    fn on_validate_app_menu_command(&self, _callback: Box<dyn FnMut(&dyn Action) -> bool>) {}

    fn thermal_state(&self) -> ThermalState {
        ThermalState::Nominal
    }

    fn on_thermal_state_change(&self, _callback: Box<dyn FnMut()>) {}

    fn compositor_name(&self) -> &'static str {
        "iOS"
    }

    fn app_path(&self) -> Result<PathBuf> {
        Err(anyhow::anyhow!("app_path is not available in the PoC"))
    }

    fn path_for_auxiliary_executable(&self, _name: &str) -> Result<PathBuf> {
        Err(anyhow::anyhow!(
            "path_for_auxiliary_executable is not available on iOS"
        ))
    }

    fn set_cursor_style(&self, _style: CursorStyle) {}

    fn hide_cursor_until_mouse_moves(&self) {
        // Fork-required drift method; iOS has no cursor.
        self.cursor_visible.set(false);
    }

    fn is_cursor_visible(&self) -> bool {
        self.cursor_visible.get()
    }

    fn should_auto_hide_scrollbars(&self) -> bool {
        true
    }

    fn read_from_clipboard(&self) -> Option<ClipboardItem> {
        None
    }

    fn write_to_clipboard(&self, _item: ClipboardItem) {}

    fn write_credentials(&self, _url: &str, _username: &str, _password: &[u8]) -> Task<Result<()>> {
        Task::ready(Err(anyhow::anyhow!(
            "credential storage is not available in the PoC"
        )))
    }

    fn read_credentials(&self, _url: &str) -> Task<Result<Option<(String, Vec<u8>)>>> {
        Task::ready(Ok(None))
    }

    fn delete_credentials(&self, _url: &str) -> Task<Result<()>> {
        Task::ready(Err(anyhow::anyhow!(
            "credential storage is not available in the PoC"
        )))
    }

    fn keyboard_layout(&self) -> Box<dyn PlatformKeyboardLayout> {
        Box::new(IosKeyboardLayout)
    }

    fn keyboard_mapper(&self) -> Rc<dyn PlatformKeyboardMapper> {
        Rc::new(DummyKeyboardMapper)
    }

    fn on_keyboard_layout_change(&self, _callback: Box<dyn FnMut()>) {}
}
