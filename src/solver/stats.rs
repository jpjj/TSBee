use crate::penalties::candidates::held_karp::HeldKarpResult;

/// A struct to hold the statistics of the solver
#[derive(Clone)]
pub struct Stats {
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub time_taken: chrono::Duration,
    pub iterations: u64,
    pub iterations_since_last_improvement: u64,
    pub held_karp_result: Option<HeldKarpResult>,
}

impl Stats {
    pub(super) fn new() -> Self {
        let start_time = chrono::Utc::now();
        let time_taken = chrono::Duration::zero();
        let iterations = 0;
        let iterations_since_last_improvement = 0;
        Self {
            start_time,
            time_taken,
            iterations,
            iterations_since_last_improvement,
            held_karp_result: None,
        }
    }

    pub(super) fn reset(&mut self) {
        self.start_time = chrono::Utc::now();
        self.time_taken = chrono::Duration::zero();
        self.iterations = 0;
        self.iterations_since_last_improvement = 0;
    }
}
