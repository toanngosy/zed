//! Inertial (fling) scrolling for touch drags: velocity capture + decay.
//!
//! When a finger drags and lifts, native platforms keep the content gliding
//! with a decelerating velocity ("momentum" / "fling"). Without it a chart pan
//! stops dead the instant the finger leaves the glass and feels sluggish.
//!
//! Two cooperating pieces, both platform-agnostic and time-driven by
//! [`std::time::Instant`] (iOS has a real monotonic clock — this crate is
//! `target_os = "ios"` only):
//!
//! - [`VelocityTracker`] records recent drag samples and estimates the release
//!   velocity when the finger lifts.
//! - [`MomentumScroller`] takes that velocity and yields decelerating
//!   per-frame displacements ([`MomentumScroller::step`]) until it settles.
//!
//! The iOS backend seeds the scroller on finger-lift and pumps [`step`] once
//! per `CADisplayLink` tick, applying each displacement as a synthetic
//! drag-continuation move (see [`crate::touch`]). We wrote this from scratch
//! against our own traits; the `references/gpui-mobile` module is the design
//! oracle only (it emits `ScrollWheel`, which our chart maps to *zoom* — so we
//! deliberately drive the *drag* channel instead, see [`crate::touch`]).
//!
//! [`step`]: MomentumScroller::step

// Rust guideline compliant 2026-02-21

use std::time::Instant;

/// Per-millisecond velocity decay. iOS `UIScrollView`'s `.normal`
/// deceleration is `0.998`/ms; at 60 fps (dt≈16.6 ms) that is ≈3.3% loss per
/// frame, so a ~2000 px/s fling glides ~2 s before settling — the native feel.
const DECELERATION_RATE: f32 = 0.998;

/// Below this speed (logical px/s) a fling is considered finished.
const MIN_VELOCITY: f32 = 30.0;

/// Upper bound (logical px/s) on a captured fling velocity. A hard flick on a
/// modern phone easily exceeds 12000 px/s; clamp so a near-zero sample dt can't
/// synthesize an absurd velocity.
const MAX_VELOCITY: f32 = 16000.0;

/// Capacity of the recent-sample ring used for velocity estimation.
const MAX_SAMPLES: usize = 16;

/// Only samples newer than this (seconds) feed the velocity fit, so a slow
/// drag start cannot dilute the fast release that actually determines the fling.
const VELOCITY_WINDOW_SECS: f64 = 0.10;

/// Fewer samples than this in the window → no meaningful velocity.
const MIN_SAMPLES_FOR_VELOCITY: usize = 2;

/// Cap on a single frame's `dt` (seconds) so a suspend/GC stall cannot teleport
/// the content on the next `step`.
const MAX_STEP_DT_SECS: f32 = 0.033;

/// A recorded drag sample (logical px + capture time).
#[derive(Clone, Copy, Debug)]
struct Sample {
    x: f32,
    y: f32,
    time: Instant,
}

/// Records recent drag positions and estimates the finger-lift velocity.
///
/// Call [`record`](Self::record) on every touch-move, then [`velocity`](Self::velocity)
/// on finger-up to obtain the fling velocity in logical px/s.
#[derive(Debug)]
pub struct VelocityTracker {
    samples: [Option<Sample>; MAX_SAMPLES],
    /// Next write index into the ring.
    index: usize,
}

impl Default for VelocityTracker {
    fn default() -> Self {
        Self {
            samples: [None; MAX_SAMPLES],
            index: 0,
        }
    }
}

impl VelocityTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a drag position (logical px). Call on every move.
    pub fn record(&mut self, x: f32, y: f32) {
        self.samples[self.index] = Some(Sample {
            x,
            y,
            time: Instant::now(),
        });
        self.index = (self.index + 1) % MAX_SAMPLES;
    }

    /// Estimate the release velocity `(vx, vy)` in logical px/s.
    ///
    /// Uses a recency-weighted least-squares fit over samples within
    /// [`VELOCITY_WINDOW_SECS`] (3+ samples) so the estimate reflects the fast
    /// end of the gesture; falls back to a first/last difference for 2 samples.
    /// Returns `(0.0, 0.0)` when too few recent samples exist.
    pub fn velocity(&self) -> (f32, f32) {
        let now = Instant::now();
        let mut recent: Vec<Sample> = self
            .samples
            .iter()
            .flatten()
            .copied()
            .filter(|s| now.duration_since(s.time).as_secs_f64() <= VELOCITY_WINDOW_SECS)
            .collect();

        if recent.len() < MIN_SAMPLES_FOR_VELOCITY {
            return (0.0, 0.0);
        }
        recent.sort_by(|a, b| a.time.cmp(&b.time));

        if recent.len() >= 3 {
            let (vx, vy) = weighted_velocity(&recent);
            return (clamp_velocity(vx), clamp_velocity(vy));
        }

        let (first, last) = (recent[0], recent[recent.len() - 1]);
        let dt = last.time.duration_since(first.time).as_secs_f64();
        if dt < 1e-6 {
            return (0.0, 0.0);
        }
        let vx = ((last.x - first.x) as f64 / dt) as f32;
        let vy = ((last.y - first.y) as f64 / dt) as f32;
        (clamp_velocity(vx), clamp_velocity(vy))
    }

    /// Clear all samples for a fresh gesture.
    pub fn reset(&mut self) {
        self.samples = [None; MAX_SAMPLES];
        self.index = 0;
    }
}

/// Recency-weighted least-squares velocity: weight `e^(2i/n)` biases the fit
/// toward the newest (fastest) samples. Falls back to a plain difference if the
/// normal equations are degenerate.
fn weighted_velocity(samples: &[Sample]) -> (f32, f32) {
    let n = samples.len();
    let t0 = samples[0].time;

    let mut sum_w = 0.0_f64;
    let mut sum_wt = 0.0_f64;
    let mut sum_wt2 = 0.0_f64;
    let mut sum_wx = 0.0_f64;
    let mut sum_wy = 0.0_f64;
    let mut sum_wtx = 0.0_f64;
    let mut sum_wty = 0.0_f64;

    for (i, s) in samples.iter().enumerate() {
        let t = s.time.duration_since(t0).as_secs_f64();
        let w = (2.0 * i as f64 / n as f64).exp();
        sum_w += w;
        sum_wt += w * t;
        sum_wt2 += w * t * t;
        sum_wx += w * s.x as f64;
        sum_wy += w * s.y as f64;
        sum_wtx += w * t * s.x as f64;
        sum_wty += w * t * s.y as f64;
    }

    let denom = sum_w * sum_wt2 - sum_wt * sum_wt;
    if denom.abs() < 1e-12 {
        let (first, last) = (samples[0], samples[n - 1]);
        let dt = last.time.duration_since(first.time).as_secs_f64();
        if dt < 1e-6 {
            return (0.0, 0.0);
        }
        return (
            ((last.x - first.x) as f64 / dt) as f32,
            ((last.y - first.y) as f64 / dt) as f32,
        );
    }

    let vx = (sum_w * sum_wtx - sum_wt * sum_wx) / denom;
    let vy = (sum_w * sum_wty - sum_wt * sum_wy) / denom;
    (vx as f32, vy as f32)
}

fn clamp_velocity(v: f32) -> f32 {
    v.clamp(-MAX_VELOCITY, MAX_VELOCITY)
}

/// One decelerating step's displacement (logical px), applied by the caller as
/// a drag-continuation move.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MomentumStep {
    /// Displacement along X since the previous step.
    pub dx: f32,
    /// Displacement along Y since the previous step.
    pub dy: f32,
}

/// Produces decelerating displacements from a release velocity.
///
/// Seed with [`fling`](Self::fling), then call [`step`](Self::step) once per
/// frame: it returns `Some(MomentumStep)` while gliding and `None` once the
/// velocity falls below [`MIN_VELOCITY`].
#[derive(Debug)]
pub struct MomentumScroller {
    vx: f32,
    vy: f32,
    active: bool,
    last_time: Instant,
}

impl Default for MomentumScroller {
    fn default() -> Self {
        Self {
            vx: 0.0,
            vy: 0.0,
            active: false,
            last_time: Instant::now(),
        }
    }
}

impl MomentumScroller {
    /// Create an idle scroller.
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a fling from a release velocity (logical px/s). A speed below
    /// [`MIN_VELOCITY`] leaves the scroller idle (no glide).
    pub fn fling(&mut self, vx: f32, vy: f32) {
        if (vx * vx + vy * vy).sqrt() < MIN_VELOCITY {
            self.active = false;
            return;
        }
        self.vx = vx;
        self.vy = vy;
        self.active = true;
        self.last_time = Instant::now();
    }

    /// Advance one frame, returning the displacement to apply, or `None` when
    /// the fling has settled.
    ///
    /// Displacement uses the analytical integral of `v·rᵗ` over the frame
    /// (`v·(r^Δt−1)/ln r`) rather than `v·Δt`, which avoids the high-velocity
    /// overshoot that makes the first frames feel jumpy.
    pub fn step(&mut self) -> Option<MomentumStep> {
        if !self.active {
            return None;
        }

        let now = Instant::now();
        let dt = (now.duration_since(self.last_time).as_secs_f64() as f32).min(MAX_STEP_DT_SECS);
        self.last_time = now;
        if dt < 1e-6 {
            return None;
        }

        let dt_ms = dt * 1000.0;
        let decay = DECELERATION_RATE.powf(dt_ms);
        let ln_r = DECELERATION_RATE.ln();
        // ∫₀^{Δt_ms} v·rᵗ dt = v·(r^Δt_ms − 1)/ln(r); /1000 converts ms→s.
        let disp_factor = if ln_r.abs() > 1e-9 {
            (decay - 1.0) / (ln_r * 1000.0)
        } else {
            dt
        };

        let dx = self.vx * disp_factor;
        let dy = self.vy * disp_factor;
        self.vx *= decay;
        self.vy *= decay;

        if (self.vx * self.vx + self.vy * self.vy).sqrt() < MIN_VELOCITY {
            self.active = false;
            // Drop a final sub-pixel tail so the caller's finalize isn't a jump.
            if dx.abs() < 0.1 && dy.abs() < 0.1 {
                return None;
            }
        }
        Some(MomentumStep { dx, dy })
    }

    /// Stop any active fling (e.g. a new touch landed).
    pub fn cancel(&mut self) {
        self.active = false;
        self.vx = 0.0;
        self.vy = 0.0;
    }

    /// Whether a fling is currently gliding.
    pub fn is_active(&self) -> bool {
        self.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn velocity_needs_two_samples() {
        let mut t = VelocityTracker::new();
        assert_eq!(t.velocity(), (0.0, 0.0));
        t.record(10.0, 20.0);
        assert_eq!(t.velocity(), (0.0, 0.0));
    }

    #[test]
    fn velocity_positive_for_downward_drag() {
        let mut t = VelocityTracker::new();
        for i in 0..8 {
            t.record(100.0, 100.0 + i as f32 * 6.0);
            sleep(Duration::from_millis(4));
        }
        let (_vx, vy) = t.velocity();
        // sleep is imprecise on CI; assert direction + a loose magnitude floor.
        assert!(vy > 50.0, "vy={vy} should be clearly positive");
    }

    #[test]
    fn velocity_is_clamped() {
        let mut t = VelocityTracker::new();
        t.record(0.0, 0.0);
        t.record(9000.0, 9000.0);
        sleep(Duration::from_micros(50));
        t.record(18000.0, 18000.0);
        let (vx, vy) = t.velocity();
        assert!(vx.abs() <= MAX_VELOCITY && vy.abs() <= MAX_VELOCITY);
    }

    #[test]
    fn reset_clears_samples() {
        let mut t = VelocityTracker::new();
        t.record(0.0, 0.0);
        sleep(Duration::from_millis(4));
        t.record(50.0, 50.0);
        t.reset();
        assert_eq!(t.velocity(), (0.0, 0.0));
    }

    #[test]
    fn slow_fling_does_not_start() {
        let mut s = MomentumScroller::new();
        s.fling(5.0, 5.0);
        assert!(!s.is_active());
        assert!(s.step().is_none());
    }

    #[test]
    fn fling_decelerates_and_settles() {
        let mut s = MomentumScroller::new();
        s.fling(0.0, 2000.0);
        assert!(s.is_active());

        let mut steps: Vec<f32> = Vec::new();
        for _ in 0..1000 {
            sleep(Duration::from_millis(16));
            match s.step() {
                Some(step) => {
                    assert!(step.dy >= 0.0);
                    steps.push(step.dy);
                }
                None => break,
            }
        }
        assert!(!s.is_active());
        assert!(
            steps.len() > 10,
            "expected many glide frames, got {}",
            steps.len()
        );
        let q = steps.len() / 4;
        if q > 0 {
            let head: f32 = steps[..q].iter().sum::<f32>() / q as f32;
            let tail: f32 = steps[steps.len() - q..].iter().sum::<f32>() / q as f32;
            assert!(head > tail, "should decelerate: head={head} tail={tail}");
        }
    }

    #[test]
    fn cancel_stops_glide() {
        let mut s = MomentumScroller::new();
        s.fling(0.0, 3000.0);
        s.cancel();
        assert!(!s.is_active());
        assert!(s.step().is_none());
    }
}
