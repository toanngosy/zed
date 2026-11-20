//! Two-finger pinch geometry of the web touch stream.
//!
//! Pure bookkeeping, host-testable: which touch pointers are down (in arrival
//! order, so the pinch pair is stable) and the centroid, separation and
//! per-axis spread of the pair that a [`PinchEvent`] reports.

use gpui::{PinchEvent, PinchSpread, Pixels, Point, Size, TouchPhase, point, px, size};

/// Touch pointers still down, in arrival order; the first two are the pinch
/// pair.
#[derive(Debug, Default)]
pub(crate) struct ActiveTouches(Vec<(i32, Point<Pixels>)>);

impl ActiveTouches {
    /// Insert or update a pointer's element-local position.
    pub(crate) fn upsert(&mut self, id: i32, pos: Point<Pixels>) {
        match self.0.iter_mut().find(|(tid, _)| *tid == id) {
            Some(entry) => entry.1 = pos,
            None => self.0.push((id, pos)),
        }
    }

    /// Stop tracking a pointer; `true` when it was one of the pinch pair.
    pub(crate) fn remove(&mut self, id: i32) -> bool {
        let pair_finger = self.0.iter().take(2).any(|(tid, _)| *tid == id);
        self.0.retain(|(tid, _)| *tid != id);
        pair_finger
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn get(&self, id: i32) -> Option<Point<Pixels>> {
        self.0
            .iter()
            .find(|(tid, _)| *tid == id)
            .map(|(_, pos)| *pos)
    }

    pub(crate) fn ids(&self) -> impl Iterator<Item = i32> + '_ {
        self.0.iter().map(|(id, _)| *id)
    }

    /// Geometry of the pinch pair, or `None` with fewer than two touches.
    pub(crate) fn pinch_sample(&self) -> Option<PinchSample> {
        match self.0.as_slice() {
            [(_, a), (_, b), ..] => Some(PinchSample::of(*a, *b)),
            _ => None,
        }
    }
}

/// Centroid, separation and per-axis spread of the two pinch fingers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PinchSample {
    pub(crate) center: Point<Pixels>,
    dist: f32,
    spread: Size<Pixels>,
}

impl PinchSample {
    fn of(a: Point<Pixels>, b: Point<Pixels>) -> Self {
        let (ax, ay) = (f32::from(a.x), f32::from(a.y));
        let (bx, by) = (f32::from(b.x), f32::from(b.y));
        let (dx, dy) = (bx - ax, by - ay);
        Self {
            center: point(px((ax + bx) / 2.0), px((ay + by) / 2.0)),
            dist: (dx * dx + dy * dy).sqrt(),
            spread: size(px(dx.abs()), px(dy.abs())),
        }
    }

    /// The pinch event at this sample, relative to `prev`.
    pub(crate) fn event(self, prev: Self, phase: TouchPhase) -> PinchEvent {
        let delta = if prev.dist > f32::EPSILON {
            self.dist / prev.dist - 1.0
        } else {
            0.0
        };
        PinchEvent {
            position: self.center,
            delta,
            modifiers: Default::default(),
            phase,
            spread: Some(PinchSpread {
                current: self.spread,
                previous: prev.spread,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f32, y: f32) -> Point<Pixels> {
        point(px(x), px(y))
    }

    #[test]
    fn pinch_event_reports_per_axis_spread() {
        let mut touches = ActiveTouches::default();
        touches.upsert(7, p(100.0, 100.0));
        touches.upsert(9, p(200.0, 140.0));
        let start = touches.pinch_sample().expect("two touches pinch");

        touches.upsert(9, p(260.0, 140.0));
        let moved = touches.pinch_sample().expect("two touches pinch");
        let event = moved.event(start, TouchPhase::Moved);

        let spread = event.spread.expect("touch pinch carries a spread");
        assert_eq!(spread.previous, size(px(100.0), px(40.0)));
        assert_eq!(spread.current, size(px(160.0), px(40.0)));
        assert_eq!(event.position, p(180.0, 120.0));
        assert!(event.delta > 0.0, "spreading fingers zooms in");
    }

    #[test]
    fn crossed_fingers_keep_a_non_negative_spread() {
        let mut touches = ActiveTouches::default();
        touches.upsert(1, p(300.0, 300.0));
        touches.upsert(2, p(100.0, 250.0));
        let sample = touches.pinch_sample().expect("two touches pinch");
        let spread = sample
            .event(sample, TouchPhase::Started)
            .spread
            .expect("spread");
        assert_eq!(spread.current, size(px(200.0), px(50.0)));
    }

    #[test]
    fn pair_is_the_two_oldest_touches() {
        let mut touches = ActiveTouches::default();
        touches.upsert(5, p(0.0, 0.0));
        touches.upsert(3, p(100.0, 0.0));
        touches.upsert(4, p(500.0, 500.0));
        assert_eq!(touches.pinch_sample().map(|s| s.center), Some(p(50.0, 0.0)));
        assert_eq!(touches.ids().collect::<Vec<_>>(), [5, 3, 4]);
        assert_eq!(touches.get(3), Some(p(100.0, 0.0)));

        assert!(!touches.remove(4), "a third finger is not in the pair");
        assert!(touches.remove(5), "the oldest finger is in the pair");
        assert_eq!(touches.len(), 1);
        assert_eq!(touches.pinch_sample(), None);
    }
}
