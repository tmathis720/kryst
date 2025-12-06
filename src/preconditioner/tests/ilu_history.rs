use crate::algebra::prelude::*;
use crate::preconditioner::stats::{ParIluHistory, ParIluIterSample};
use crate::utils::monitor::{Event, Monitor};
use std::sync::Mutex;

struct TestMonitor(Mutex<Vec<&'static str>>);

impl Monitor for TestMonitor {
    fn on_event(&self, ev: Event<'_>) {
        let mut v = self.0.lock().unwrap();
        match ev {
            Event::IluSetupBegin { .. } => v.push("begin"),
            Event::IluSetupIter { .. } => v.push("iter"),
            Event::IluSetupEnd { .. } => v.push("end"),
        }
    }
}

#[test]
    fn history_and_monitor_basic() {
        let mut h = ParIluHistory::with_capacity(2);
        h.push(ParIluIterSample {
            iter: 1,
            residual: 0.3,
        });
    assert_eq!(h.as_slice().len(), 1);

    let m = TestMonitor(Mutex::new(Vec::new()));
    m.on_event(Event::IluSetupBegin { opts_hash: 0 });
    m.on_event(Event::IluSetupIter {
        sample: &h.as_slice()[0],
    });
    m.on_event(Event::IluSetupEnd {
        iters: 1,
        converged: true,
        setup_time_s: R::default(),
    });
    let events = m.0.lock().unwrap();
    assert_eq!(&events[..], ["begin", "iter", "end"]);
}
