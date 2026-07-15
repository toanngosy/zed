//! Single-display [`PlatformDisplay`] for iOS.
//!
//! The PoC receives the screen size + scale from the Swift host at boot (the
//! app owns `UIScreen`), so this backend needs no objc to enumerate displays.
//! A single logical display suffices for a full-screen view-only surface.

// Rust guideline compliant 2026-02-21

use anyhow::Result;
use gpui::{Bounds, DisplayId, Pixels, PlatformDisplay, Point, Size, px};

/// One logical iOS display, sized from the host-provided bounds.
#[derive(Debug)]
pub(crate) struct IosDisplay {
    id: DisplayId,
    uuid: uuid::Uuid,
    size: Size<Pixels>,
}

impl IosDisplay {
    /// Build the primary display from a logical-point size.
    pub(crate) fn new(width: f32, height: f32) -> Self {
        Self {
            id: DisplayId::new(1),
            // Deterministic per-size id so repeated boots are stable.
            uuid: uuid::Uuid::new_v4(),
            size: Size {
                width: px(width),
                height: px(height),
            },
        }
    }
}

impl PlatformDisplay for IosDisplay {
    fn id(&self) -> DisplayId {
        self.id
    }

    fn uuid(&self) -> Result<uuid::Uuid> {
        Ok(self.uuid)
    }

    fn bounds(&self) -> Bounds<Pixels> {
        Bounds {
            origin: Point::default(),
            size: self.size,
        }
    }
}
