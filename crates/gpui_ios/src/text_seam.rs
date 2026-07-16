//! The iOS text-input platform seam: soft-keyboard toggle + keystroke forwarding.
//!
//! GPUI routes text input through [`PlatformWindow::set_input_handler`] /
//! [`PlatformWindow::take_input_handler`]: a focused text field installs a
//! [`PlatformInputHandler`]; when focus leaves, GPUI takes it back. This module
//! owns that handler for the iOS backend and turns host keystrokes (forwarded
//! from a Swift `UIKeyInput` responder) into [`PlatformInputHandler`] edits.
//!
//! # The keyboard is PULLED, not pushed
//!
//! GPUI calls `take_input_handler` then `set_input_handler` on EVERY frame it
//! draws — it borrows the handler out to paint the text field and puts it back.
//! Calling `becomeFirstResponder` / `resignFirstResponder` on each of those
//! would thrash the software keyboard once per frame. Instead
//! [`TextInputState::set_handler`] / [`TextInputState::take_handler`] only flip a
//! [`wants_keyboard`](TextInputState::wants_keyboard) flag; the Swift host polls
//! it once per `CADisplayLink` tick and reconciles the first-responder state
//! *only when it changes*. A field that stays focused re-installs its handler in
//! the same frame it was taken, before the host next polls, so a steady focus
//! never flickers the keyboard.
//!
//! This mirrors the `gpui_web` backend, where the identical set/take pair
//! toggles a hidden `<input>`'s `readonly` / `inputmode` idempotently.
//!
//! # Deferred (issue #1453)
//!
//! Scope here is basic ASCII insert/delete + keyboard show/hide. Marked-text /
//! IME composition (CJK, dead keys, predictive candidates) and
//! grapheme-cluster-aware backspace are intentionally out of scope. The seam
//! records the caret rect via [`TextInputState::set_caret`] (from
//! `update_ime_position`) so the host can frame its hidden responder at the
//! caret and a future IME can anchor its candidate window there, but basic mode
//! consumes it no further.

// Rust guideline compliant 2026-02-21

// This module's production consumers (`window.rs`, `ffi.rs`) are
// `cfg(target_os = "ios")`; on a non-iOS host the only consumer is the test
// module below. Silence dead-code lints for the host lib-only build (so
// `clippy -D warnings` stays green there) WITHOUT masking real dead code on iOS
// or under `cargo test`.
#![cfg_attr(all(not(target_os = "ios"), not(test)), allow(dead_code))]

use std::cell::{Cell, RefCell};
use std::ops::Range;

use gpui::{Bounds, Pixels, PlatformInputHandler, UTF16Selection};

/// Owns the active [`PlatformInputHandler`] and the derived soft-keyboard state
/// for the one iOS window.
///
/// Main-thread-only (single [`RefCell`] / [`Cell`] cells, no `Send`/`Sync`):
/// every caller — GPUI's draw loop and the FFI text entry points — runs on the
/// iOS main thread.
pub(crate) struct TextInputState {
    /// The handler GPUI installs for the focused text field, if any.
    handler: RefCell<Option<PlatformInputHandler>>,
    /// Whether a text field currently wants the soft keyboard. Pulled by the
    /// host each frame; see the module docs for why this is a flag, not a call.
    wants_keyboard: Cell<bool>,
    /// The last caret rect reported by `update_ime_position` (logical pixels).
    caret: Cell<Bounds<Pixels>>,
}

impl TextInputState {
    /// Create an empty seam: no handler, keyboard hidden, caret at the origin.
    pub(crate) fn new() -> Self {
        Self {
            handler: RefCell::new(None),
            wants_keyboard: Cell::new(false),
            caret: Cell::new(Bounds::default()),
        }
    }

    /// Install the handler for a newly-focused text field and request the
    /// keyboard.
    pub(crate) fn set_handler(&self, handler: PlatformInputHandler) {
        *self.handler.borrow_mut() = Some(handler);
        self.wants_keyboard.set(true);
    }

    /// Take the handler back and drop the keyboard request.
    ///
    /// GPUI calls this every frame; a still-focused field re-installs via
    /// [`set_handler`](Self::set_handler) within the same frame, so
    /// [`wants_keyboard`](Self::wants_keyboard) is back to `true` before the host
    /// next polls. A genuine blur leaves it `false`, and the host resigns the
    /// first responder.
    pub(crate) fn take_handler(&self) -> Option<PlatformInputHandler> {
        self.wants_keyboard.set(false);
        self.handler.borrow_mut().take()
    }

    /// Whether a text field currently wants the soft keyboard shown.
    pub(crate) fn wants_keyboard(&self) -> bool {
        self.wants_keyboard.get()
    }

    /// Record the caret rect (logical pixels) from `update_ime_position`.
    pub(crate) fn set_caret(&self, caret: Bounds<Pixels>) {
        self.caret.set(caret);
    }

    /// The last caret rect reported (logical pixels), or the origin if none.
    pub(crate) fn caret(&self) -> Bounds<Pixels> {
        self.caret.get()
    }

    /// Insert host-provided text at the caret (UIKeyInput `insertText:`).
    ///
    /// A no-op when no field owns the handler.
    pub(crate) fn insert(&self, text: &str) {
        self.with_handler(|handler| handler.replace_text_in_range(None, text));
    }

    /// Delete the selection, or one UTF-16 unit before the caret (UIKeyInput
    /// `deleteBackward`).
    ///
    /// A no-op when no field owns the handler or nothing precedes the caret.
    pub(crate) fn delete_backward(&self) {
        self.with_handler(|handler| {
            if let Some(range) = backspace_range(handler.selected_text_range(true)) {
                handler.replace_text_in_range(Some(range), "");
            }
        });
    }

    /// Run `f` with the handler taken OUT of its cell and put back afterward.
    ///
    /// Forwarding re-enters GPUI (`replace_text_in_range` drives an app update),
    /// which can itself call [`take_handler`](Self::take_handler) on this same
    /// cell; holding the handler outside the cell for the duration keeps that
    /// re-entrant borrow from double-borrow-panicking — the take-invoke-restore
    /// discipline the frame/touch callbacks use. If a re-entrant `set_handler`
    /// installed a fresh handler while ours was out, the fresh one wins and ours
    /// is dropped.
    fn with_handler<R>(&self, f: impl FnOnce(&mut PlatformInputHandler) -> R) -> Option<R> {
        let mut handler = self.handler.borrow_mut().take()?;
        let result = f(&mut handler);
        let mut slot = self.handler.borrow_mut();
        if slot.is_none() {
            *slot = Some(handler);
        }
        Some(result)
    }
}

/// The UTF-16 range a backspace should delete, or `None` when nothing precedes
/// the caret.
///
/// A non-empty selection is deleted whole; otherwise one UTF-16 code unit before
/// the caret is removed. Single-code-unit deletion is correct for ASCII/BMP
/// text; grapheme-cluster-aware deletion (combining marks, emoji ZWJ sequences)
/// is deferred with IME support (issue #1453).
fn backspace_range(selection: Option<UTF16Selection>) -> Option<Range<usize>> {
    let range = selection?.range;
    if !range.is_empty() {
        Some(range)
    } else if range.start > 0 {
        Some(range.start - 1..range.start)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caret_at(pos: usize) -> UTF16Selection {
        UTF16Selection {
            range: pos..pos,
            reversed: false,
        }
    }

    #[test]
    fn backspace_deletes_one_unit_before_caret() {
        assert_eq!(backspace_range(Some(caret_at(3))), Some(2..3));
    }

    #[test]
    fn backspace_at_document_start_is_noop() {
        assert_eq!(backspace_range(Some(caret_at(0))), None);
    }

    #[test]
    fn backspace_deletes_a_whole_selection() {
        let sel = UTF16Selection {
            range: 1..4,
            reversed: false,
        };
        assert_eq!(backspace_range(Some(sel)), Some(1..4));
    }

    #[test]
    fn backspace_without_a_handler_selection_is_noop() {
        assert_eq!(backspace_range(None), None);
    }

    // --- End-to-end headless test: a real PlatformInputHandler over a test
    // EntityInputHandler, driven through the seam's insert/delete + keyboard flag.

    use gpui::{
        Bounds, Context, ElementInputHandler, EntityInputHandler, IntoElement,
        PlatformInputHandler, Point, Render, Size, TestAppContext, UTF16Selection, Window, div, px,
    };
    use std::ops::Range;

    /// Minimal text field: a UTF-8 buffer with a caret. Test input is ASCII, so
    /// UTF-16 offsets equal byte offsets throughout.
    struct TestField {
        text: String,
        selection: Range<usize>,
    }

    impl Render for TestField {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div()
        }
    }

    impl EntityInputHandler for TestField {
        fn text_for_range(
            &mut self,
            range: Range<usize>,
            _adjusted: &mut Option<Range<usize>>,
            _window: &mut Window,
            _cx: &mut Context<Self>,
        ) -> Option<String> {
            self.text.get(range).map(str::to_owned)
        }

        fn selected_text_range(
            &mut self,
            _ignore_disabled_input: bool,
            _window: &mut Window,
            _cx: &mut Context<Self>,
        ) -> Option<UTF16Selection> {
            Some(UTF16Selection {
                range: self.selection.clone(),
                reversed: false,
            })
        }

        fn marked_text_range(
            &self,
            _window: &mut Window,
            _cx: &mut Context<Self>,
        ) -> Option<Range<usize>> {
            None
        }

        fn unmark_text(&mut self, _window: &mut Window, _cx: &mut Context<Self>) {}

        fn replace_text_in_range(
            &mut self,
            range: Option<Range<usize>>,
            text: &str,
            _window: &mut Window,
            cx: &mut Context<Self>,
        ) {
            let range = range.unwrap_or_else(|| self.selection.clone());
            self.text.replace_range(range.clone(), text);
            let caret = range.start + text.len();
            self.selection = caret..caret;
            cx.notify();
        }

        fn replace_and_mark_text_in_range(
            &mut self,
            _range: Option<Range<usize>>,
            _new_text: &str,
            _new_selected_range: Option<Range<usize>>,
            _window: &mut Window,
            _cx: &mut Context<Self>,
        ) {
        }

        fn bounds_for_range(
            &mut self,
            _range_utf16: Range<usize>,
            _element_bounds: Bounds<Pixels>,
            _window: &mut Window,
            _cx: &mut Context<Self>,
        ) -> Option<Bounds<Pixels>> {
            None
        }

        fn character_index_for_point(
            &mut self,
            _point: Point<Pixels>,
            _window: &mut Window,
            _cx: &mut Context<Self>,
        ) -> Option<usize> {
            None
        }
    }

    #[gpui::test]
    fn keystrokes_reach_the_installed_input_handler(cx: &mut TestAppContext) {
        let (field, cx) = cx.add_window_view(|_window, _cx| TestField {
            text: String::new(),
            selection: 0..0,
        });

        // Build the exact PlatformInputHandler GPUI would hand the backend for
        // this focused field, then install it through the seam.
        let platform_handler = cx.update(|window, app| {
            PlatformInputHandler::new(
                window.to_async(app),
                Box::new(ElementInputHandler::new(Bounds::default(), field.clone())),
            )
        });

        let state = TextInputState::new();
        assert!(!state.wants_keyboard(), "no field focused yet");

        state.set_handler(platform_handler);
        assert!(state.wants_keyboard(), "focus requests the keyboard");

        // `update_ime_position` round-trips the caret rect for the host to frame
        // its hidden responder.
        let caret = Bounds {
            origin: Point {
                x: px(10.0),
                y: px(20.0),
            },
            size: Size {
                width: px(2.0),
                height: px(16.0),
            },
        };
        state.set_caret(caret);
        assert_eq!(state.caret(), caret, "caret rect round-trips");

        // Type "hi!", then backspace the "!".
        state.insert("hi");
        state.insert("!");
        state.delete_backward();

        let typed = cx.update(|_window, app| field.read(app).text.clone());
        assert_eq!(typed, "hi", "insert + delete flowed to the input handler");

        // Blur: the handler is returned and the keyboard is no longer wanted.
        let returned = state.take_handler();
        assert!(returned.is_some(), "blur hands the handler back");
        assert!(!state.wants_keyboard(), "blur drops the keyboard request");
    }
}
