//! GCD-backed [`PlatformDispatcher`] for iOS.
//!
//! libdispatch (GCD) is identical on iOS and macOS, so this mirrors
//! `gpui_macos::MacDispatcher` almost verbatim: the fork's
//! [`RunnableVariant`] is leaked to a raw context pointer, handed to a GCD
//! queue, and reconstituted inside a C `trampoline`. The two differences from
//! macOS: the main-thread check uses `pthread_main_np` (no AppKit), and
//! `spawn_realtime` is a plain thread (iOS grants no real-time thread policy to
//! app processes for a view-only surface).

// Rust guideline compliant 2026-02-21

use dispatch2::{DispatchQueue, DispatchQueueGlobalPriority, DispatchTime, GlobalQueueIdentifier};
use gpui::{PlatformDispatcher, Priority, RunnableVariant};
use std::{ffi::c_void, ptr::NonNull, time::Duration};

/// Unit dispatcher; all state lives in GCD's global/main queues.
pub(crate) struct IosDispatcher;

impl IosDispatcher {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl PlatformDispatcher for IosDispatcher {
    fn is_main_thread(&self) -> bool {
        // SAFETY: `pthread_main_np` is always safe to call and returns non-zero
        // on the process main thread. Avoids pulling AppKit/`NSThread` on iOS.
        unsafe { libc::pthread_main_np() != 0 }
    }

    fn dispatch(&self, runnable: RunnableVariant, priority: Priority) {
        let context = runnable.into_raw().as_ptr() as *mut c_void;
        let queue_priority = match priority {
            Priority::RealtimeAudio => {
                panic!("RealtimeAudio priority should use spawn_realtime, not dispatch")
            }
            Priority::High => DispatchQueueGlobalPriority::High,
            Priority::Medium => DispatchQueueGlobalPriority::Default,
            Priority::Low => DispatchQueueGlobalPriority::Low,
        };
        // SAFETY: `context` is a live leaked `RunnableVariant`; `trampoline`
        // reclaims and runs exactly once. GCD guarantees a single invocation.
        unsafe {
            DispatchQueue::global_queue(GlobalQueueIdentifier::Priority(queue_priority))
                .exec_async_f(context, trampoline);
        }
    }

    fn dispatch_on_main_thread(&self, runnable: RunnableVariant, _priority: Priority) {
        let context = runnable.into_raw().as_ptr() as *mut c_void;
        // SAFETY: see `dispatch`; the main queue interleaves with the app's
        // `CADisplayLink` on the same run loop.
        unsafe {
            DispatchQueue::main().exec_async_f(context, trampoline);
        }
    }

    fn dispatch_after(&self, duration: Duration, runnable: RunnableVariant) {
        let context = runnable.into_raw().as_ptr() as *mut c_void;
        let queue = DispatchQueue::global_queue(GlobalQueueIdentifier::Priority(
            DispatchQueueGlobalPriority::High,
        ));
        let when = DispatchTime::NOW.time(duration.as_nanos() as i64);
        // SAFETY: see `dispatch`.
        unsafe {
            DispatchQueue::exec_after_f(when, &queue, context, trampoline);
        }
    }

    fn spawn_realtime(&self, f: Box<dyn FnOnce() + Send>) {
        std::thread::spawn(move || f());
    }
}

/// GCD callback: reconstitute the leaked [`RunnableVariant`] and run it once.
extern "C" fn trampoline(context: *mut c_void) {
    // SAFETY: `context` is the exact pointer produced by `into_raw` in one of
    // the dispatch methods above; GCD delivers it once, so reclaiming ownership
    // here is sound.
    let runnable = unsafe { RunnableVariant::from_raw(NonNull::new_unchecked(context as *mut ())) };
    runnable.run();
}
