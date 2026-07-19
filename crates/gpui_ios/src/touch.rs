//! Multi-touch aggregation: UIKit touches → GPUI [`PlatformInput`].
//!
//! The Swift host forwards every `UITouch` (each phase, each finger) across the
//! C seam as `(pointer_id, phase, x, y)` in logical pixels; [`TouchAggregator`]
//! turns that raw stream into the same cooked [`PlatformInput`] events the
//! mature backends emit, so all gesture *semantics* stay in shared consumer
//! code (the chart's own tap / pan / long-press / double-tap recognizer and its
//! `on_pinch` handler). Nothing chart-specific lives here.
//!
//! # Phase ABI (the C seam's private contract)
//!
//! The `phase` argument is **our own** small-integer convention, NOT raw
//! `UITouchPhase`. The Swift shim MUST translate UIKit phases to these values
//! (a shim forwarding raw `UITouchPhase` codes — `.stationary = 2`,
//! `.cancelled = 4` — would silently mis-map input):
//!
//! | `phase` | meaning        | UIKit source                         |
//! |---------|----------------|--------------------------------------|
//! | `0`     | [`PHASE_BEGAN`]     | `touchesBegan`                  |
//! | `1`     | [`PHASE_MOVED`]     | `touchesMoved` (+ `.stationary`)|
//! | `2`     | [`PHASE_ENDED`]     | `touchesEnded`                  |
//! | `3`     | [`PHASE_CANCELLED`] | `touchesCancelled`              |
//!
//! `pointer_id` is a stable per-finger identity for the lifetime of one touch
//! (the shim uses the `UITouch` object identity). It is what lets this
//! aggregator track two concurrent fingers for pinch — the previous single-`i32`
//! ABI carried no identity and could not. An unknown `phase` is ignored.
//!
//! # Gesture mapping (web-parity + iOS momentum)
//!
//! - **One finger** → `MouseDown` / `MouseMove{is_touch}` / `MouseUp` **AND** a
//!   parallel `ScrollWheel{touch_phase: Moved}` on every drag step. A one-finger
//!   drag now drives BOTH channels: the chart consumes the `MouseMove` to pan (it
//!   ignores touch-phase `ScrollWheel`, so it never zooms on a finger drag), while
//!   gpui scroll containers (bottom sheets, lists) consume the `ScrollWheel` to
//!   scroll. This additive scroll channel is why sheets/lists scroll on touch; we
//!   keep `MouseMove` because the consumer's recognizer classifies tap / pan /
//!   long-press / double-tap off it — the same path the web backend drives. The
//!   scroll delta is `pos - previous_pos` (natural direction, content tracks the
//!   finger), mirroring the gpui-mobile oracle's `src/ios/window.rs`, and is
//!   dominant-axis-locked per drag (see [`AxisLock`]) so a diagonal drag can't
//!   scroll a nested carousel and its parent sheet at once.
//! - **Two fingers** → [`PlatformInput::Pinch`] (`Started`/`Moved`/`Ended`),
//!   mirroring `gpui_web`'s 2-touch aggregation. The second finger's press is
//!   suppressed (no `MouseDown`); the consumer cancels the first finger's
//!   pending press on `Started`. No `ScrollWheel` is emitted while pinching.
//! - **Momentum** (iOS-only, not on web yet): a fast one-finger lift seeds a
//!   [`MomentumScroller`]; [`TouchAggregator::pump_momentum`] then continues the
//!   fling on BOTH channels — decaying synthetic `MouseMove`s (chart pan) and
//!   `ScrollWheel`s (scroll deceleration) — then a final `MouseUp`.

// Rust guideline compliant 2026-02-21

use gpui::{
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, PinchEvent, Pixels, PlatformInput,
    Point, ScrollDelta, ScrollWheelEvent, Size, TouchPhase, point, px,
};

use crate::momentum::{MomentumScroller, MomentumStep, VelocityTracker};

/// Finger touched down (`touchesBegan`).
pub const PHASE_BEGAN: i32 = 0;
/// Finger moved or stayed put (`touchesMoved` / `.stationary`).
pub const PHASE_MOVED: i32 = 1;
/// Finger lifted (`touchesEnded`).
pub const PHASE_ENDED: i32 = 2;
/// Touch interrupted by the system (`touchesCancelled`).
pub const PHASE_CANCELLED: i32 = 3;

/// Minimum drag displacement (logical px) along the dominant axis before a
/// one-finger scroll commits to that axis.
///
/// Small enough that a deliberate swipe locks almost immediately, large enough
/// that the first jittery pixels of a press don't pick a spurious axis. Tuned by
/// feel on-device; mirrored by the `gpui_web` backend so touch and web agree.
const AXIS_LOCK_THRESHOLD_PX: f32 = 8.0;

/// The axis a one-finger scroll has committed to for the rest of a drag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LockedAxis {
    Horizontal,
    Vertical,
}

impl LockedAxis {
    /// The delta with the component orthogonal to this axis zeroed.
    fn project(self, delta: Point<Pixels>) -> Point<Pixels> {
        match self {
            LockedAxis::Horizontal => point(delta.x, px(0.0)),
            LockedAxis::Vertical => point(px(0.0), delta.y),
        }
    }
}

/// Per-gesture dominant-axis lock for the additive one-finger scroll channel.
///
/// A one-finger drag emits a `ScrollWheel` on top of its `MouseMove`. Without a
/// lock, a diagonal drag over a horizontally-scrolling carousel nested inside a
/// vertically-scrolling sheet scrolls BOTH at once — the containers fight and
/// the content jitters (issue #1567). GPUI's `div` scroll neither restricts to
/// an axis nor stops propagation, and the nested pair can't coordinate via
/// per-element flags, so the fix lives here at the emitter: accumulate the drag
/// displacement and, once it clears [`AXIS_LOCK_THRESHOLD_PX`], lock to the
/// larger-magnitude axis and zero the minor component of every later scroll
/// delta. The `MouseMove` channel (chart pan) stays full 2D and is unaffected.
#[derive(Clone, Copy, Debug, Default)]
struct AxisLock {
    /// Signed drag displacement since the gesture began (logical px).
    accum: Point<Pixels>,
    /// The committed axis, or `None` until the threshold is cleared.
    locked: Option<LockedAxis>,
}

impl AxisLock {
    /// Clear all state for a fresh gesture.
    fn reset(&mut self) {
        *self = Self::default();
    }

    /// The committed axis, if the drag has locked yet.
    fn locked(&self) -> Option<LockedAxis> {
        self.locked
    }

    /// Feed one raw per-step scroll delta; returns the delta to actually emit,
    /// with the minor axis zeroed once the drag has locked.
    ///
    /// Accumulates the displacement first, so the drag locks to whichever axis
    /// dominates the motion from the gesture's start — not just the latest step.
    fn apply(&mut self, delta: Point<Pixels>) -> Point<Pixels> {
        self.accum = point(self.accum.x + delta.x, self.accum.y + delta.y);
        if self.locked.is_none() {
            let (ax, ay) = (f32::from(self.accum.x).abs(), f32::from(self.accum.y).abs());
            if ax.max(ay) >= AXIS_LOCK_THRESHOLD_PX {
                self.locked = Some(if ax >= ay {
                    LockedAxis::Horizontal
                } else {
                    LockedAxis::Vertical
                });
            }
        }
        match self.locked {
            Some(axis) => axis.project(delta),
            None => delta,
        }
    }
}

/// Aggregates the raw multi-touch stream into cooked [`PlatformInput`] events.
///
/// Main-thread only (like the whole FFI seam); holds no `unsafe` `Send`/`Sync`.
#[derive(Debug)]
pub struct TouchAggregator {
    /// Active fingers in arrival order; the first two define pinch geometry.
    active: Vec<(u64, Point<Pixels>)>,
    /// Finger separation on the previous pinch sample; `None` when not pinching.
    pinch_prev_dist: Option<f32>,
    /// A pinch happened during the current touch sequence — suppresses a stray
    /// one-finger pan / momentum from the leftover finger until all lift.
    pinch_occurred: bool,
    velocity: VelocityTracker,
    /// Dominant-axis lock for the current one-finger drag's scroll channel.
    axis_lock: AxisLock,
    momentum: MomentumScroller,
    /// Synthetic drag position while a fling glides (logical px).
    momentum_pos: Point<Pixels>,
    /// A one-finger `MouseUp` is deferred while momentum glides the drag.
    momentum_open: bool,
    /// Axis the fling is locked to, carried from the drag so the glide scrolls
    /// only along the committed axis. `None` once the drag never locked.
    momentum_axis: Option<LockedAxis>,
}

impl Default for TouchAggregator {
    fn default() -> Self {
        Self {
            active: Vec::new(),
            pinch_prev_dist: None,
            pinch_occurred: false,
            velocity: VelocityTracker::new(),
            axis_lock: AxisLock::default(),
            momentum: MomentumScroller::new(),
            momentum_pos: Point::default(),
            momentum_open: false,
            momentum_axis: None,
        }
    }
}

impl TouchAggregator {
    /// Create an aggregator with no active touches.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one forwarded touch; returns the [`PlatformInput`]s to dispatch.
    ///
    /// `pos` is in logical pixels. An unrecognized `phase` yields no events.
    pub fn on_touch(&mut self, id: u64, phase: i32, pos: Point<Pixels>) -> Vec<PlatformInput> {
        match phase {
            PHASE_BEGAN => self.on_began(id, pos),
            PHASE_MOVED => self.on_moved(id, pos),
            PHASE_ENDED => self.on_ended(id, pos, false),
            PHASE_CANCELLED => self.on_ended(id, pos, true),
            _ => Vec::new(),
        }
    }

    /// Advance an in-flight fling by one frame; returns the drag-continuation
    /// event, the finalizing `MouseUp`, or nothing when idle.
    ///
    /// Positions are clamped inside `bounds` so the synthetic move (and the
    /// final up) stay over the consumer's hit area — GPUI mouse dispatch is
    /// hover-gated, so an off-window position would be silently dropped and wedge
    /// the deferred drag.
    pub fn pump_momentum(&mut self, bounds: Size<Pixels>) -> Vec<PlatformInput> {
        if !self.momentum_open {
            return Vec::new();
        }
        match self.momentum.step() {
            Some(MomentumStep { dx, dy }) => {
                let np = clamp_to_bounds(
                    point(self.momentum_pos.x + px(dx), self.momentum_pos.y + px(dy)),
                    bounds,
                );
                self.momentum_pos = np;
                // Drive both channels during the glide: `MouseMove` continues the
                // chart pan (full 2D), `ScrollWheel` decelerates any gpui scroll
                // container. The scroll delta is the per-frame displacement (same
                // natural direction as the drag), but projected onto the axis the
                // drag locked to so a diagonal flick doesn't fling both nested
                // containers at once (issue #1567).
                let step = point(px(dx), px(dy));
                let scroll = self.momentum_axis.map_or(step, |axis| axis.project(step));
                vec![scroll_wheel(np, scroll, TouchPhase::Moved), mouse_move(np)]
            }
            None => {
                self.momentum_open = false;
                vec![mouse_up(self.momentum_pos)]
            }
        }
    }

    // ── phase handlers ──────────────────────────────────────────────────

    fn on_began(&mut self, id: u64, pos: Point<Pixels>) -> Vec<PlatformInput> {
        let mut out = Vec::new();
        // A fresh touch stops any glide; finalize the deferred drag first.
        if self.momentum_open {
            out.push(mouse_up(self.momentum_pos));
            self.momentum_open = false;
            self.momentum.cancel();
            self.momentum_axis = None;
        }
        self.touch_upsert(id, pos);

        if self.active.len() >= 2 {
            // Second finger → pinch. Suppress this finger's press. A pinch
            // invalidates any single-finger axis lock the first finger built up.
            self.axis_lock.reset();
            if let Some((center, dist)) = self.pinch_geometry() {
                self.pinch_prev_dist = Some(dist);
                self.pinch_occurred = true;
                out.push(pinch(center, 0.0, TouchPhase::Started));
            }
        } else {
            // First finger → a single-finger press; the recognizer classifies it.
            // Start the drag with a clean axis lock (dominant axis not yet chosen).
            self.velocity.reset();
            self.axis_lock.reset();
            self.velocity.record(f32::from(pos.x), f32::from(pos.y));
            out.push(PlatformInput::MouseDown(MouseDownEvent {
                button: MouseButton::Left,
                position: pos,
                modifiers: Default::default(),
                click_count: 1,
                first_mouse: true,
            }));
        }
        out
    }

    fn on_moved(&mut self, id: u64, pos: Point<Pixels>) -> Vec<PlatformInput> {
        let mut out = Vec::new();
        // Capture THIS finger's prior tracked position before the upsert so a
        // one-finger drag can also emit a `ScrollWheel` delta (`pos - prev`).
        let prev = self
            .active
            .iter()
            .find(|(tid, _)| *tid == id)
            .map(|(_, p)| *p);
        self.touch_upsert(id, pos);

        if self.active.len() >= 2 {
            // Two fingers → incremental pinch; suppress the pan.
            if let Some((center, dist)) = self.pinch_geometry() {
                if let Some(prev) = self.pinch_prev_dist {
                    if prev > f32::EPSILON {
                        out.push(pinch(center, dist / prev - 1.0, TouchPhase::Moved));
                    }
                }
                self.pinch_prev_dist = Some(dist);
            }
        } else {
            // Lone finger. Only a genuine drag (not the leftover finger after a
            // pinch) tracks velocity and drives the scroll channel.
            if !self.pinch_occurred {
                self.velocity.record(f32::from(pos.x), f32::from(pos.y));
                // Additive scroll channel: a one-finger drag also scrolls gpui
                // scroll containers (sheets/lists). The chart ignores touch-phase
                // `ScrollWheel` and pans off the `MouseMove` below instead. The
                // delta is dominant-axis-locked so a diagonal drag doesn't scroll
                // a nested carousel and its parent sheet at once (issue #1567).
                if let Some(prev) = prev {
                    let scroll = self.axis_lock.apply(point(pos.x - prev.x, pos.y - prev.y));
                    out.push(scroll_wheel(pos, scroll, TouchPhase::Moved));
                }
            }
            out.push(mouse_move(pos));
        }
        out
    }

    fn on_ended(&mut self, id: u64, pos: Point<Pixels>, cancelled: bool) -> Vec<PlatformInput> {
        let mut out = Vec::new();
        let was_pinching = self.active.len() >= 2;
        self.touch_remove(id);

        if was_pinching {
            // A finger left a pinch. End it once fewer than two remain; the
            // suppressed finger never became a press, so emit no `MouseUp`.
            if self.active.len() < 2 {
                self.pinch_prev_dist = None;
                out.push(pinch(pos, 0.0, TouchPhase::Ended));
            }
        } else if !cancelled && !self.pinch_occurred {
            // Clean one-finger lift: fling on a fast release, else end the drag.
            let (vx, vy) = self.velocity.velocity();
            self.momentum.fling(vx, vy);
            if self.momentum.is_active() {
                self.momentum_pos = pos;
                self.momentum_open = true;
                // Carry the drag's axis lock into the fling so the glide stays on
                // the committed axis; the drag lock itself resets below.
                self.momentum_axis = self.axis_lock.locked();
            } else {
                out.push(mouse_up(pos));
            }
            self.velocity.reset();
            self.axis_lock.reset();
        } else {
            // Cancelled, or the leftover finger after a pinch: end plainly. Also
            // fold a cancel arriving mid-glide back into a clean finalize.
            if self.momentum_open {
                out.push(mouse_up(self.momentum_pos));
                self.momentum_open = false;
                self.momentum.cancel();
                self.momentum_axis = None;
            } else {
                out.push(mouse_up(pos));
            }
            self.velocity.reset();
            self.axis_lock.reset();
        }

        if self.active.is_empty() {
            self.pinch_occurred = false;
        }
        out
    }

    // ── touch bookkeeping ───────────────────────────────────────────────

    fn touch_upsert(&mut self, id: u64, pos: Point<Pixels>) {
        if let Some(entry) = self.active.iter_mut().find(|(tid, _)| *tid == id) {
            entry.1 = pos;
        } else {
            self.active.push((id, pos));
        }
    }

    fn touch_remove(&mut self, id: u64) {
        self.active.retain(|(tid, _)| *tid != id);
    }

    /// Pinch centroid + finger separation from the first two active touches, or
    /// `None` with fewer than two.
    fn pinch_geometry(&self) -> Option<(Point<Pixels>, f32)> {
        if self.active.len() < 2 {
            return None;
        }
        let (a, b) = (self.active[0].1, self.active[1].1);
        let (ax, ay) = (f32::from(a.x), f32::from(a.y));
        let (bx, by) = (f32::from(b.x), f32::from(b.y));
        let center = point(px((ax + bx) / 2.0), px((ay + by) / 2.0));
        let (dx, dy) = (bx - ax, by - ay);
        Some((center, (dx * dx + dy * dy).sqrt()))
    }
}

fn mouse_move(pos: Point<Pixels>) -> PlatformInput {
    PlatformInput::MouseMove(MouseMoveEvent {
        position: pos,
        pressed_button: Some(MouseButton::Left),
        modifiers: Default::default(),
        is_touch: true,
    })
}

fn mouse_up(pos: Point<Pixels>) -> PlatformInput {
    PlatformInput::MouseUp(MouseUpEvent {
        button: MouseButton::Left,
        position: pos,
        modifiers: Default::default(),
        click_count: 1,
    })
}

/// A pixel-precise, touch-sourced `ScrollWheel` (`is_touch: true`), so a consumer
/// that maps device scroll to zoom (the chart) can ignore finger-drag scrolls
/// while scroll containers still consume them. `touch_phase` alone can't express
/// this — a device wheel also reports `Moved`.
fn scroll_wheel(position: Point<Pixels>, delta: Point<Pixels>, phase: TouchPhase) -> PlatformInput {
    PlatformInput::ScrollWheel(ScrollWheelEvent {
        position,
        delta: ScrollDelta::Pixels(delta),
        modifiers: Default::default(),
        touch_phase: phase,
        is_touch: true,
    })
}

fn pinch(position: Point<Pixels>, delta: f32, phase: TouchPhase) -> PlatformInput {
    PlatformInput::Pinch(PinchEvent {
        position,
        delta,
        modifiers: Default::default(),
        phase,
    })
}

/// Clamp a point strictly inside `[0, w] × [0, h]` (half-pixel margin) so a
/// hover-gated consumer still receives the event.
fn clamp_to_bounds(p: Point<Pixels>, bounds: Size<Pixels>) -> Point<Pixels> {
    let w = f32::from(bounds.width);
    let h = f32::from(bounds.height);
    let cx = f32::from(p.x).clamp(0.5, (w - 0.5).max(0.5));
    let cy = f32::from(p.y).clamp(0.5, (h - 0.5).max(0.5));
    point(px(cx), px(cy))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f32, y: f32) -> Point<Pixels> {
        point(px(x), px(y))
    }

    /// Whether `input` is a `Pinch` in the given phase (`TouchPhase` is not
    /// `PartialEq`, so we match structurally).
    fn is_pinch(input: &PlatformInput, phase: TouchPhase) -> bool {
        matches!(input, PlatformInput::Pinch(e)
            if std::mem::discriminant(&e.phase) == std::mem::discriminant(&phase))
    }

    /// The pixel scroll delta if `input` is a `ScrollWheel` in the given phase.
    fn scroll_delta(input: &PlatformInput, phase: TouchPhase) -> Option<Point<Pixels>> {
        match input {
            PlatformInput::ScrollWheel(e)
                if std::mem::discriminant(&e.touch_phase) == std::mem::discriminant(&phase) =>
            {
                match e.delta {
                    ScrollDelta::Pixels(d) => Some(d),
                    ScrollDelta::Lines(_) => None,
                }
            }
            _ => None,
        }
    }

    #[test]
    fn one_finger_down_move_drives_both_scroll_and_move() {
        let mut agg = TouchAggregator::new();

        let down = agg.on_touch(1, PHASE_BEGAN, p(10.0, 10.0));
        assert!(matches!(down.as_slice(), [PlatformInput::MouseDown(_)]));

        // A one-finger drag now yields a ScrollWheel (for scroll containers)
        // ALONGSIDE the MouseMove (for the chart pan / recognizer).
        let mv = agg.on_touch(1, PHASE_MOVED, p(40.0, 12.0));
        assert!(matches!(
            mv.as_slice(),
            [PlatformInput::ScrollWheel(_), PlatformInput::MouseMove(m)] if m.is_touch
        ));
        // Raw delta = pos - previous_pos = (40,12) - (10,10) = (30, 2). The drag
        // is dominantly horizontal (30 >> 2) and clears the lock threshold on this
        // first step, so the dominant-axis lock zeroes the minor (vertical)
        // component: emitted delta is (30, 0), natural sign on the locked axis.
        let delta = scroll_delta(&mv[0], TouchPhase::Moved).expect("scroll wheel moved");
        assert_eq!(f32::from(delta.x), 30.0);
        assert_eq!(f32::from(delta.y), 0.0);

        // Clear the captured velocity so the lift is a deterministic no-fling
        // release (the sub-microsecond spacing between the synthetic samples
        // otherwise races the velocity window); the fling path has its own test.
        agg.velocity.reset();
        let up = agg.on_touch(1, PHASE_ENDED, p(40.0, 12.0));
        assert!(matches!(up.as_slice(), [PlatformInput::MouseUp(_)]));
        assert!(!agg.momentum_open);
    }

    #[test]
    fn drag_locks_to_vertical_then_ignores_horizontal_wiggle() {
        let mut agg = TouchAggregator::new();
        agg.on_touch(1, PHASE_BEGAN, p(100.0, 100.0));

        // First step is dominantly vertical (2 vs 10) and clears the threshold →
        // lock vertical, zeroing the minor (horizontal) scroll component.
        let mv1 = agg.on_touch(1, PHASE_MOVED, p(102.0, 110.0));
        let d1 = scroll_delta(&mv1[0], TouchPhase::Moved).expect("scroll wheel moved");
        assert_eq!(
            f32::from(d1.x),
            0.0,
            "horizontal component zeroed once locked"
        );
        assert_eq!(f32::from(d1.y), 10.0);
        assert_eq!(agg.axis_lock.locked(), Some(LockedAxis::Vertical));

        // A later horizontal wiggle stays zeroed on X — the lock holds all drag.
        let mv2 = agg.on_touch(1, PHASE_MOVED, p(140.0, 112.0));
        let d2 = scroll_delta(&mv2[0], TouchPhase::Moved).expect("scroll wheel moved");
        assert_eq!(
            f32::from(d2.x),
            0.0,
            "horizontal wiggle suppressed after lock"
        );
        assert_eq!(f32::from(d2.y), 2.0);
    }

    #[test]
    fn drag_locks_to_horizontal_then_ignores_vertical_wiggle() {
        let mut agg = TouchAggregator::new();
        agg.on_touch(1, PHASE_BEGAN, p(100.0, 100.0));

        // First step is dominantly horizontal (12 vs 1) → lock horizontal.
        let mv1 = agg.on_touch(1, PHASE_MOVED, p(112.0, 101.0));
        let d1 = scroll_delta(&mv1[0], TouchPhase::Moved).expect("scroll wheel moved");
        assert_eq!(f32::from(d1.x), 12.0);
        assert_eq!(
            f32::from(d1.y),
            0.0,
            "vertical component zeroed once locked"
        );
        assert_eq!(agg.axis_lock.locked(), Some(LockedAxis::Horizontal));

        // A later vertical wiggle stays zeroed on Y — the lock holds all drag.
        let mv2 = agg.on_touch(1, PHASE_MOVED, p(114.0, 140.0));
        let d2 = scroll_delta(&mv2[0], TouchPhase::Moved).expect("scroll wheel moved");
        assert_eq!(f32::from(d2.x), 2.0);
        assert_eq!(
            f32::from(d2.y),
            0.0,
            "vertical wiggle suppressed after lock"
        );
    }

    #[test]
    fn sub_threshold_drag_does_not_lock_yet() {
        let mut agg = TouchAggregator::new();
        agg.on_touch(1, PHASE_BEGAN, p(100.0, 100.0));

        // A tiny move under the threshold picks no axis: the full 2D scroll delta
        // passes through unchanged and the lock stays open.
        let mv = agg.on_touch(1, PHASE_MOVED, p(103.0, 102.0));
        let d = scroll_delta(&mv[0], TouchPhase::Moved).expect("scroll wheel moved");
        assert_eq!(f32::from(d.x), 3.0);
        assert_eq!(f32::from(d.y), 2.0);
        assert_eq!(
            agg.axis_lock.locked(),
            None,
            "below threshold must not lock"
        );
    }

    #[test]
    fn momentum_glide_respects_axis_lock() {
        use std::thread::sleep;
        use std::time::Duration;

        let mut agg = TouchAggregator::new();
        // Seed a diagonal fling but with the drag locked to the vertical axis, as
        // a vertical drag would leave it on lift. The glide must scroll only
        // vertically — the horizontal fling component is projected away.
        agg.momentum.fling(2000.0, 3000.0);
        agg.momentum_pos = p(100.0, 400.0);
        agg.momentum_open = true;
        agg.momentum_axis = Some(LockedAxis::Vertical);
        let bounds = Size {
            width: px(400.0),
            height: px(800.0),
        };

        sleep(Duration::from_millis(8));
        let step = agg.pump_momentum(bounds);
        let delta = scroll_delta(&step[0], TouchPhase::Moved).expect("scroll wheel moved");
        assert_eq!(
            f32::from(delta.x),
            0.0,
            "locked-vertical fling must not scroll horizontally"
        );
        assert!(f32::from(delta.y) > 0.0, "vertical fling still glides");
    }

    #[test]
    fn two_finger_move_emits_no_stray_scroll() {
        let mut agg = TouchAggregator::new();
        agg.on_touch(1, PHASE_BEGAN, p(100.0, 100.0));
        agg.on_touch(2, PHASE_BEGAN, p(200.0, 100.0));

        // Spreading fingers must be Pinch only — no ScrollWheel or MouseMove.
        let moved = agg.on_touch(2, PHASE_MOVED, p(260.0, 100.0));
        assert_eq!(moved.len(), 1);
        assert!(is_pinch(&moved[0], TouchPhase::Moved));
        assert!(scroll_delta(&moved[0], TouchPhase::Moved).is_none());
    }

    #[test]
    fn fast_lift_pumps_decaying_scroll_then_mouse_up() {
        use std::thread::sleep;
        use std::time::Duration;

        let mut agg = TouchAggregator::new();
        // Seed a fast vertical fling directly on the scroller so the test does
        // not depend on wall-clock velocity capture (that path is time-driven
        // and covered by momentum.rs's own tests).
        agg.momentum.fling(0.0, 3000.0);
        agg.momentum_pos = p(100.0, 400.0);
        agg.momentum_open = true;
        let bounds = Size {
            width: px(400.0),
            height: px(800.0),
        };

        // A gliding frame drives BOTH channels: ScrollWheel then the drag move.
        sleep(Duration::from_millis(8));
        let step = agg.pump_momentum(bounds);
        assert!(matches!(
            step.as_slice(),
            [PlatformInput::ScrollWheel(_), PlatformInput::MouseMove(m)] if m.is_touch
        ));
        assert!(scroll_delta(&step[0], TouchPhase::Moved).is_some());

        // Once the fling settles, the deferred one-finger lift finalizes as a
        // single MouseUp and the momentum channel closes.
        agg.momentum.cancel();
        let end = agg.pump_momentum(bounds);
        assert!(matches!(end.as_slice(), [PlatformInput::MouseUp(_)]));
        assert!(!agg.momentum_open);
    }

    #[test]
    fn second_finger_starts_pinch_and_suppresses_press() {
        let mut agg = TouchAggregator::new();
        agg.on_touch(1, PHASE_BEGAN, p(100.0, 100.0));

        let began2 = agg.on_touch(2, PHASE_BEGAN, p(200.0, 100.0));
        assert_eq!(began2.len(), 1);
        assert!(is_pinch(&began2[0], TouchPhase::Started));

        // A move with two fingers apart → Pinch Moved, never a MouseMove.
        let moved = agg.on_touch(2, PHASE_MOVED, p(260.0, 100.0));
        assert_eq!(moved.len(), 1);
        assert!(is_pinch(&moved[0], TouchPhase::Moved));
        if let PlatformInput::Pinch(e) = &moved[0] {
            assert!(e.delta > 0.0, "spreading fingers zooms in (delta>0)");
        }
    }

    #[test]
    fn pinch_delta_negative_when_pinching_together() {
        let mut agg = TouchAggregator::new();
        agg.on_touch(1, PHASE_BEGAN, p(100.0, 100.0));
        agg.on_touch(2, PHASE_BEGAN, p(300.0, 100.0));
        let moved = agg.on_touch(2, PHASE_MOVED, p(200.0, 100.0));
        if let PlatformInput::Pinch(e) = &moved[0] {
            assert!(e.delta < 0.0, "closing fingers zooms out (delta<0)");
        } else {
            panic!("expected pinch move");
        }
    }

    #[test]
    fn lifting_one_finger_ends_pinch_without_mouse_up() {
        let mut agg = TouchAggregator::new();
        agg.on_touch(1, PHASE_BEGAN, p(100.0, 100.0));
        agg.on_touch(2, PHASE_BEGAN, p(200.0, 100.0));

        let end = agg.on_touch(2, PHASE_ENDED, p(200.0, 100.0));
        assert_eq!(end.len(), 1);
        assert!(is_pinch(&end[0], TouchPhase::Ended));

        // The leftover first finger must NOT resume as a pan (its press was
        // suppressed) — a move produces a MouseMove but no fling on lift.
        let up = agg.on_touch(1, PHASE_ENDED, p(100.0, 100.0));
        assert!(matches!(up.as_slice(), [PlatformInput::MouseUp(_)]));
        assert!(!agg.momentum_open);
    }

    #[test]
    fn cancel_ends_a_one_finger_drag() {
        let mut agg = TouchAggregator::new();
        agg.on_touch(1, PHASE_BEGAN, p(10.0, 10.0));
        let out = agg.on_touch(1, PHASE_CANCELLED, p(30.0, 10.0));
        assert!(matches!(out.as_slice(), [PlatformInput::MouseUp(_)]));
    }

    #[test]
    fn pump_is_noop_without_a_fling() {
        let mut agg = TouchAggregator::new();
        assert!(
            agg.pump_momentum(Size {
                width: px(400.0),
                height: px(800.0)
            })
            .is_empty()
        );
    }

    #[test]
    fn clamp_keeps_position_inside_bounds() {
        let b = Size {
            width: px(400.0),
            height: px(800.0),
        };
        let inside = clamp_to_bounds(p(-50.0, 5000.0), b);
        assert!(f32::from(inside.x) >= 0.5 && f32::from(inside.x) <= 399.5);
        assert!(f32::from(inside.y) >= 0.5 && f32::from(inside.y) <= 799.5);
    }
}
