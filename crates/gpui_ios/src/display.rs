//! Single-display [`PlatformDisplay`] for iOS.
//!
//! The PoC receives the screen size + scale from the Swift host at boot (the
//! app owns `UIScreen`), so this backend needs no objc to enumerate displays.
//! A single logical display suffices for a full-screen view-only surface.

// Rust guideline compliant 2026-02-21

use anyhow::Result;
use gpui::{Bounds, DisplayId, Pixels, PlatformDisplay, Point, Size, px};

/// Stable identity for the single logical iOS display.
///
/// GPUI keys displays by their `uuid`, so a fixed value keeps the identity
/// constant across boots and rotations. The spike used `Uuid::new_v4()`, which
/// minted a fresh id every boot — the opposite of the "deterministic id" the
/// comment claimed; this constant delivers the stability that was intended.
const IOS_DISPLAY_UUID: uuid::Uuid =
    uuid::Uuid::from_u128(0x105a_fe10_105a_fe10_105a_fe10_105a_fe10);

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
            uuid: IOS_DISPLAY_UUID,
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
