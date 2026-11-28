//! Persistent workers for ready, owned executor batches.
//! Go multiplexes executor goroutines onto reusable runtime threads. This
//! queue likewise keeps CPU workers alive, parking them when no work is ready.
//! Blocking input pipelines run on separate lanes; jobs move their buffers
//! and return ownership at a completion barrier.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

type LaneTask = Box<dyn FnOnce() + Send + 'static>;

/// One reusable process-wide lane set. The width is part of the registry key
/// so changing the session concurrency still gets a matching lane set.
struct SharedLanePool {
    sender: std::sync::mpsc::SyncSender<LaneTask>,
    in_use: AtomicBool,
}

#[derive(Hash, PartialEq, Eq)]
struct LanePoolKey {
    name: &'static str,
    workers: usize,
}

fn lane_registry() -> &'static Mutex<HashMap<LanePoolKey, Vec<Arc<SharedLanePool>>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<LanePoolKey, Vec<Arc<SharedLanePool>>>>> =
        OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn shared_lane_pool(name: &'static str, workers: usize) -> Arc<SharedLanePool> {
    let mut registry = lane_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let pools = registry.entry(LanePoolKey { name, workers }).or_default();
    if let Some(pool) = pools.iter().find(|pool| {
        pool.in_use
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }) {
        return Arc::clone(pool);
    }
    let pool = create_lane_pool(name, workers);
    pools.push(Arc::clone(&pool));
    pool
}

fn create_lane_pool(name: &'static str, workers: usize) -> Arc<SharedLanePool> {
    let (sender, receiver) = std::sync::mpsc::sync_channel::<LaneTask>(workers);
    let receiver = Arc::new(Mutex::new(receiver));
    for _ in 0..workers {
        let receiver = Arc::clone(&receiver);
        std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || loop {
                let task = receiver
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .recv();
                let Ok(task) = task else { break };
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task));
            })
            .expect("spawn persistent exec lane");
    }
    Arc::new(SharedLanePool {
        sender,
        in_use: AtomicBool::new(true),
    })
}

struct LaneActivity {
    state: Mutex<LaneActivityState>,
    done: Condvar,
}

struct LaneActivityState {
    in_flight: usize,
    closed: bool,
}

/// A per-executor view of a process-wide registry of reusable lanes for work
/// that may block on external I/O. Go's index-join workers are goroutines
/// reused for every task; idle sets are returned to the registry after Close,
/// while overlapping executors receive separate sets to preserve concurrency.
pub struct LanePool {
    shared: Arc<SharedLanePool>,
    activity: Arc<LaneActivity>,
    width: usize,
}

impl LanePool {
    /// Gets `concurrency` persistent lanes and bounds this executor's
    /// in-flight work to the same count, matching Go's buffered worker
    /// channel. Native workers are created once per concurrently active set
    /// and reused by later executors.
    pub fn new(name: &'static str, concurrency: usize) -> Self {
        let workers = concurrency.max(1);
        Self::with_lanes(shared_lane_pool(name, workers), workers)
    }

    /// Creates reader-owned workers that are released when the pool is dropped.
    /// Use for jobs whose changing width would otherwise keep idle workers in the registry.
    pub fn new_dedicated(name: &'static str, concurrency: usize) -> Self {
        let workers = concurrency.max(1);
        Self::with_lanes(create_lane_pool(name, workers), workers)
    }

    fn with_lanes(shared: Arc<SharedLanePool>, width: usize) -> Self {
        Self {
            shared,
            activity: Arc::new(LaneActivity {
                state: Mutex::new(LaneActivityState {
                    in_flight: 0,
                    closed: false,
                }),
                done: Condvar::new(),
            }),
            width,
        }
    }

    /// Queues one task for a reusable lane. Admission is bounded per
    /// executor, while idle lane sets are reused across executors.
    pub fn submit<F>(&self, task: F) -> Result<(), ()>
    where
        F: FnOnce() + Send + 'static,
    {
        {
            let mut state = self
                .activity
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            while state.in_flight >= self.width && !state.closed {
                state = self
                    .activity
                    .done
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            if state.closed {
                return Err(());
            }
            state.in_flight += 1;
        }
        let activity = Arc::clone(&self.activity);
        let result = self.shared.sender.send(Box::new(move || {
            let _guard = LaneActivityGuard { activity };
            task();
        }));
        if result.is_err() {
            let mut state = self
                .activity
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.in_flight = state.in_flight.saturating_sub(1);
            self.activity.done.notify_all();
        }
        result.map_err(|_| ())
    }

    /// Waits for every task currently admitted through this executor view.
    /// Unlike `Drop`, this keeps the reusable lane set open for a later
    /// build/restore round.
    pub fn wait(&self) {
        let mut state = self
            .activity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.in_flight != 0 {
            state = self
                .activity
                .done
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

struct LaneActivityGuard {
    activity: Arc<LaneActivity>,
}

impl Drop for LaneActivityGuard {
    fn drop(&mut self) {
        let mut state = self
            .activity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.in_flight = state.in_flight.saturating_sub(1);
        self.activity.done.notify_all();
    }
}
impl Drop for LanePool {
    fn drop(&mut self) {
        let mut state = self
            .activity
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.closed = true;
        self.activity.done.notify_all();
        while state.in_flight != 0 {
            state = self
                .activity
                .done
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        self.shared.in_use.store(false, Ordering::Release);
    }
}

/// One queued unit of work.
struct Task {
    run: Box<dyn FnOnce() + Send>,
    group: Option<Arc<AtomicBool>>,
}

/// Go's finish channel and worker wait group for one executor on the shared
/// CPU pool. Drop cancels queued work and joins evaluations already running.
/// The result channel is independent: receiving an error does not lose joins.
pub struct TaskGroup {
    pool: Arc<Shared>,
    cancelled: Arc<AtomicBool>,
    completion: Option<std::sync::mpsc::Sender<()>>,
    done: std::sync::mpsc::Receiver<()>,
}

impl TaskGroup {
    /// A task group on the process's shared worker pool.
    pub fn new() -> Self {
        Self::on_pool(Arc::clone(shared()))
    }

    fn on_pool(pool: Arc<Shared>) -> Self {
        let (completion, done) = std::sync::mpsc::channel();
        Self {
            pool,
            cancelled: Arc::new(AtomicBool::new(false)),
            completion: Some(completion),
            done,
        }
    }

    /// Runs `task` on the pool as a member of this group.
    pub fn submit(&self, task: impl FnOnce() + Send + 'static) {
        let completion = self.completion.as_ref().expect("live task group").clone();
        let cancelled = Arc::clone(&self.cancelled);
        enqueue_on(
            &self.pool,
            Task {
                run: Box::new(move || {
                    // This sender is dropped after all task captures, including
                    // on unwind. Channel EOF is the completion barrier.
                    let _completion = completion;
                    if cancelled.load(Ordering::Acquire) {
                        drop(task);
                    } else {
                        task();
                    }
                }),
                group: Some(Arc::clone(&self.cancelled)),
            },
        );
    }
}

impl Drop for TaskGroup {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        // A drained executor has no task holding its identity. Avoid touching
        // the shared queue on this usual close path. Rust's exclusive Drop
        // prevents new submissions after this observation.
        if Arc::strong_count(&self.cancelled) == 1 {
            return;
        }
        let mut cancelled = Vec::new();
        {
            let mut state = self
                .pool
                .queue
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Remove only this owner's queued work, preserving other owners'
            // order. A task already popped by a worker retains its join token.
            for _ in 0..state.pending.len() {
                let task = state.pending.pop_front().expect("queued task");
                if task
                    .group
                    .as_ref()
                    .is_some_and(|group| Arc::ptr_eq(group, &self.cancelled))
                {
                    cancelled.push(task);
                } else {
                    state.pending.push_back(task);
                }
            }
        }
        // Captures can own other executors; their Drop may acquire this queue.
        drop(cancelled);
        self.completion.take();
        let _ = self.done.recv();
    }
}

#[derive(Default)]
struct QueueState {
    pending: VecDeque<Task>,
}

#[derive(Default)]
struct Shared {
    queue: Mutex<QueueState>,
    signal: Condvar,
}

fn shared() -> &'static Arc<Shared> {
    static POOL: OnceLock<Arc<Shared>> = OnceLock::new();
    POOL.get_or_init(|| {
        let shared = Arc::new(Shared::default());
        // Go's GOMAXPROCS: one worker per core. Every task on this pool is
        // compute that runs to completion; a lane that blocks for a query's
        // lifetime (an aggregate partial lane, an index-join task draining
        // TiKV, a lookup batch) is a goroutine in Go and gets its own thread
        // here ([`spawn_lane`]), so a blocked lane never holds a core's
        // worker and the pool never starves behind one.
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(2);
        for _ in 0..workers {
            let worker_shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name("tidb-exec-pool".to_owned())
                .spawn(move || worker_loop(worker_shared))
                .expect("spawn persistent exec pool worker");
        }
        shared
    })
}

fn worker_loop(shared: Arc<Shared>) {
    loop {
        let task = {
            let mut state = shared
                .queue
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            loop {
                if let Some(task) = state.pending.pop_front() {
                    break task;
                }
                state = shared
                    .signal
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        };
        (task.run)();
    }
}

/// [`enqueue`] for out-of-crate callers.
pub fn enqueue_public(task: Box<dyn FnOnce() + Send>) {
    enqueue(task);
}

fn enqueue(task: Box<dyn FnOnce() + Send>) {
    enqueue_on(
        shared(),
        Task {
            run: task,
            group: None,
        },
    );
}

fn enqueue_on(shared: &Shared, task: Task) {
    {
        let mut state = shared
            .queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.pending.push_back(task);
    }
    shared.signal.notify_one();
}

/// Runs `task` on one of the pool's persistent workers and blocks until it
/// finishes, returning its result.
pub fn submit<F, R>(task: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let (result_tx, result_rx) = std::sync::mpsc::sync_channel::<R>(1);
    enqueue(Box::new(move || {
        // The receiver is only gone if the submitting thread died; either way
        // the value has nowhere to go.
        let _ = result_tx.send(task());
    }));
    result_rx
        .recv()
        .unwrap_or_else(|_| panic!("exec pool worker dropped the task result"))
}

/// Runs a long-lived lane on a lane thread and returns a receiver for its
/// result: Go's goroutine for a worker that blocks on a channel or on TiKV
/// for a query's lifetime. Such a lane must not occupy one of the pool's
/// per-core workers, where it would starve the short compute tasks (and
/// deadlock a core-sized pool when the tasks it waits for are queued behind
/// it). The lane starts at once, as Go's goroutine does: on an idle lane
/// thread of the same name, or on a new one when none is idle.
///
/// A lane that panics drops its result sender, so the receiver reports
/// disconnection, as a joined panicked thread reports an error.
pub fn spawn_lane<F, R>(name: &'static str, task: F) -> std::sync::mpsc::Receiver<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let (result_tx, result_rx) = std::sync::mpsc::sync_channel::<R>(1);
    spawn_lane_detached(name, move || {
        // A disconnected receiver means the caller dropped it before joining;
        // the value has nowhere to go either way.
        let _ = result_tx.send(task());
    });
    result_rx
}

/// [`spawn_lane`] for a lane that reports through its own channel.
pub fn spawn_lane_detached<F>(name: &'static str, task: F)
where
    F: FnOnce() + Send + 'static,
{
    spawn_lane_detached_with_keep_alive(name, IDLE_LANE_KEEP_ALIVE, Box::new(task));
}

/// How long a lane thread with no lane to run waits for one before it exits.
/// Go multiplexes goroutines onto threads it keeps; creating and tearing down
/// an OS thread (its stack mappings, first-touch faults and allocator cache)
/// for every lane of every statement costs far more than the lane's work in
/// short queries.
const IDLE_LANE_KEEP_ALIVE: std::time::Duration = std::time::Duration::from_secs(10);

/// An idle lane thread: the hand-off slot it waits on for its next lane.
struct IdleLane {
    id: u64,
    sender: std::sync::mpsc::SyncSender<LaneTask>,
}

/// Idle lane threads by name, the most recently idle last.
fn idle_lanes() -> &'static Mutex<HashMap<&'static str, Vec<IdleLane>>> {
    static IDLE: OnceLock<Mutex<HashMap<&'static str, Vec<IdleLane>>>> = OnceLock::new();
    IDLE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn spawn_lane_detached_with_keep_alive(
    name: &'static str,
    keep_alive: std::time::Duration,
    mut task: LaneTask,
) {
    loop {
        // The most recently idle thread is the warmest one; older ones are
        // left to retire.
        let idle = idle_lanes()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_mut(name)
            .and_then(Vec::pop);
        let Some(lane) = idle else { break };
        // An idle thread's slot is empty, so this never blocks. A thread
        // leaves the idle list before it exits, so a send fails only if
        // it died some other way; the task then goes to the next one.
        match lane.sender.send(task) {
            Ok(()) => return,
            Err(std::sync::mpsc::SendError(returned)) => task = returned,
        }
    }
    static NEXT_LANE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT_LANE_ID.fetch_add(1, Ordering::Relaxed);
    // Go's runtime aborts the process when it cannot create a thread
    // (`newosproc`); a lane that cannot start has no other home either.
    std::thread::Builder::new()
        .name(name.to_owned())
        .spawn(move || run_lane_thread(name, id, keep_alive, task))
        .expect("spawn exec lane thread");
}

/// A lane thread's life: its first lane, then each lane handed to it while
/// it is idle, until it stays idle for `keep_alive`.
///
/// Lanes share the thread one after another, so a lane must scope any
/// thread-local state it sets (restore it when it ends), as
/// `tidb_cpuprofile::labels::inherit` and the tracing span guards do.
fn run_lane_thread(name: &'static str, id: u64, keep_alive: std::time::Duration, first: LaneTask) {
    let (sender, receiver) = std::sync::mpsc::sync_channel::<LaneTask>(1);
    let mut task = first;
    loop {
        // A panicking lane is Go's crashed goroutine to its caller (the
        // panic hook has reported it and its result sender is dropped); the
        // thread stays usable for the next lane.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task));
        idle_lanes()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(name)
            .or_default()
            .push(IdleLane {
                id,
                sender: sender.clone(),
            });
        task = match receiver.recv_timeout(keep_alive) {
            Ok(task) => task,
            Err(_) => {
                let mut idle = idle_lanes()
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let lanes = idle.entry(name).or_default();
                if let Some(position) = lanes.iter().position(|lane| lane.id == id) {
                    lanes.swap_remove(position);
                    return;
                }
                drop(idle);
                // A spawner took this thread off the idle list as it timed
                // out; that lane is already sent or about to be.
                match receiver.recv() {
                    Ok(task) => task,
                    Err(_) => return,
                }
            }
        };
    }
}

/// Submits `task` without blocking and returns a receiver for its result.
/// Unlike [`submit`], the caller may keep working while the task runs.
pub fn spawn<F, R>(task: F) -> std::sync::mpsc::Receiver<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let (result_tx, result_rx) = std::sync::mpsc::sync_channel::<R>(1);
    enqueue(Box::new(move || {
        // A disconnected receiver means the caller dropped it before joining;
        // the value has nowhere to go either way.
        let _ = result_tx.send(task());
    }));
    result_rx
}

/// Whether the persistent pool exists in this process (it always does once
/// initialized; this gate lets callers fall back to a dedicated thread when
/// the pool feature is compiled out).
#[must_use]
pub fn available() -> bool {
    true
}

/// Runs one task per item across at most `concurrency` workers and returns
/// every result in submission order.
pub fn map<I, F, R>(tasks: I, concurrency: usize) -> Vec<R>
where
    I: IntoIterator<Item = F>,
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let all: Vec<F> = tasks.into_iter().collect();
    if all.len() <= 1 || concurrency <= 1 {
        return all.into_iter().map(|task| task()).collect();
    }
    let total = all.len();
    let tasks = Arc::new(Mutex::new(all.into_iter().enumerate()));
    let (result_tx, result_rx) = std::sync::mpsc::channel();
    let mut results: Vec<Option<R>> = Vec::with_capacity(total);
    results.resize_with(total, || None);

    // All input is ready. Each worker claims a whole batch, then computes
    // outside the lock; no worker waits for a producer or result consumer.
    for _ in 0..concurrency.min(total) {
        let tasks = Arc::clone(&tasks);
        let result_tx = result_tx.clone();
        enqueue(Box::new(move || loop {
            let next = tasks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .next();
            let Some((index, task)) = next else { break };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task));
            let _ = result_tx.send((index, result));
        }));
    }
    drop(result_tx);

    // Channel EOF joins every admitted worker, including on task panic.
    // Propagate the first panic only after all owned input has been released.
    let mut panic = None;
    for (index, result) in result_rx {
        match result {
            Ok(value) => results[index] = Some(value),
            Err(error) if panic.is_none() => panic = Some(error),
            Err(_) => {}
        }
    }
    if let Some(error) = panic {
        std::panic::resume_unwind(error);
    }
    results
        .into_iter()
        .map(|slot| slot.unwrap_or_else(|| panic!("missing mapped task result")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct QueuedCapture {
        pool: Arc<Shared>,
        dropped: Arc<AtomicBool>,
    }

    impl Drop for QueuedCapture {
        fn drop(&mut self) {
            assert!(
                self.pool.queue.try_lock().is_ok(),
                "capture dropped under queue lock"
            );
            self.dropped.store(true, Ordering::Release);
        }
    }

    #[test]
    fn task_group_cancels_only_its_queued_work_and_releases_captures() {
        // An isolated queue makes cancellation deterministic even when the
        // process pool has idle workers. No worker is needed for retired work.
        let pool = Arc::new(Shared::default());
        let group = TaskGroup::on_pool(Arc::clone(&pool));
        let other = TaskGroup::on_pool(Arc::clone(&pool));
        let dropped = Arc::new(AtomicBool::new(false));
        let capture = QueuedCapture {
            pool: Arc::clone(&pool),
            dropped: Arc::clone(&dropped),
        };
        group.submit(move || {
            drop(capture);
            panic!("cancelled work must not run");
        });
        let (tx, rx) = std::sync::mpsc::channel();
        for i in 0..3 {
            let tx = tx.clone();
            if i == 1 {
                other.submit(move || tx.send(i).unwrap());
            } else {
                enqueue_on(
                    &pool,
                    Task {
                        run: Box::new(move || tx.send(i).unwrap()),
                        group: None,
                    },
                );
            }
        }
        drop(group);
        assert!(dropped.load(Ordering::Acquire));
        let pending = std::mem::take(&mut pool.queue.lock().unwrap().pending);
        assert_eq!(pending.len(), 3);
        for task in pending {
            (task.run)();
        }
        drop(other);
        drop(tx);
        assert_eq!(rx.into_iter().collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn task_group_waits_for_running_work_and_unwinding_captures() {
        for panic in [false, true] {
            let pool = Arc::new(Shared::default());
            let group = TaskGroup::on_pool(Arc::clone(&pool));
            let dropped = Arc::new(AtomicBool::new(false));
            let capture = QueuedCapture {
                pool: Arc::clone(&pool),
                dropped: Arc::clone(&dropped),
            };
            let (started, starting) = std::sync::mpsc::channel();
            let (release, released) = std::sync::mpsc::channel();
            group.submit(move || {
                let _capture = capture;
                started.send(()).unwrap();
                released.recv().unwrap();
                assert!(!panic, "task panic");
            });
            let task = pool.queue.lock().unwrap().pending.pop_front().unwrap();
            let worker = std::thread::spawn(move || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(task.run))
            });
            starting.recv().unwrap();
            let (closing, closing_rx) = std::sync::mpsc::channel();
            let (closed, closed_rx) = std::sync::mpsc::channel();
            let closer = std::thread::spawn(move || {
                closing.send(()).unwrap();
                drop(group);
                closed.send(()).unwrap();
            });
            closing_rx.recv().unwrap();
            let early = closed_rx.recv_timeout(std::time::Duration::from_millis(100));
            release.send(()).unwrap();
            closed_rx.recv().unwrap();
            let captures_released_at_close = dropped.load(Ordering::Acquire);
            assert_eq!(worker.join().unwrap().is_err(), panic);
            closer.join().unwrap();
            assert!(early.is_err(), "close returned before the task completed");
            assert!(captures_released_at_close);
        }
    }

    #[test]
    fn task_group_cancels_work_claimed_but_not_started() {
        let pool = Arc::new(Shared::default());
        let group = TaskGroup::on_pool(Arc::clone(&pool));
        let cancelled = Arc::clone(&group.cancelled);
        let dropped = Arc::new(AtomicBool::new(false));
        let capture = QueuedCapture {
            pool: Arc::clone(&pool),
            dropped: Arc::clone(&dropped),
        };
        group.submit(move || {
            drop(capture);
            panic!("work claimed before close must still check cancellation");
        });
        let task = pool.queue.lock().unwrap().pending.pop_front().unwrap();
        let worker = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while !cancelled.load(Ordering::Acquire) {
                assert!(std::time::Instant::now() < deadline);
                std::thread::yield_now();
            }
            (task.run)();
        });
        drop(group);
        assert!(dropped.load(Ordering::Acquire));
        worker.join().unwrap();
    }

    #[test]
    fn submit_returns_the_task_value() {
        let value = 21 * 2;
        assert_eq!(submit(move || value), 42);
    }

    #[test]
    fn blocking_lane_workers_are_reused_across_handles() {
        // Width one makes the worker identity deterministic. A second handle
        // with the same Go-style lane shape must attach to the existing
        // process worker instead of starting another native thread.
        let (first_tx, first_rx) = std::sync::mpsc::sync_channel(1);
        let first = LanePool::new("tidb-worker-pool-reuse-test", 1);
        first
            .submit(move || first_tx.send(std::thread::current().id()).unwrap())
            .unwrap();
        drop(first);
        let first_id = first_rx.recv().unwrap();

        let (second_tx, second_rx) = std::sync::mpsc::sync_channel(1);
        let second = LanePool::new("tidb-worker-pool-reuse-test", 1);
        second
            .submit(move || second_tx.send(std::thread::current().id()).unwrap())
            .unwrap();
        drop(second);
        assert_eq!(first_id, second_rx.recv().unwrap());
    }

    /// Waits until a thread of lane `name` has finished its lane and parked.
    fn wait_until_lane_thread_is_idle(name: &'static str) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while idle_lanes()
            .lock()
            .unwrap()
            .get(name)
            .is_none_or(Vec::is_empty)
        {
            assert!(
                std::time::Instant::now() < deadline,
                "the lane thread never went idle"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn a_lane_runs_on_an_idle_lane_thread_of_its_name() {
        let name = "tidb-lane-reuse-test";
        let first = spawn_lane(name, || std::thread::current().id())
            .recv()
            .unwrap();
        wait_until_lane_thread_is_idle(name);
        let second = spawn_lane(name, || std::thread::current().id())
            .recv()
            .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn a_lane_starts_while_another_lane_of_its_name_blocks() {
        let name = "tidb-lane-concurrency-test";
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel::<()>(0);
        let blocked = spawn_lane(name, move || release_rx.recv().unwrap());
        let started = spawn_lane(name, || 7).recv_timeout(std::time::Duration::from_secs(5));
        assert_eq!(started, Ok(7));
        release_tx.send(()).unwrap();
        blocked.recv().unwrap();
    }

    #[test]
    fn a_panicking_lane_disconnects_its_result_and_keeps_its_thread() {
        let name = "tidb-lane-panic-test";
        let (thread_tx, thread_rx) = std::sync::mpsc::sync_channel(1);
        let result = spawn_lane(name, move || {
            thread_tx.send(std::thread::current().id()).unwrap();
            panic!("lane panic");
        });
        assert!(result.recv().is_err());
        let panicked_on = thread_rx.recv().unwrap();
        wait_until_lane_thread_is_idle(name);
        let next = spawn_lane(name, || std::thread::current().id())
            .recv()
            .unwrap();
        assert_eq!(next, panicked_on);
    }

    #[test]
    fn an_idle_lane_thread_exits_after_its_keep_alive() {
        struct ExitSignal(std::sync::mpsc::SyncSender<()>);
        impl Drop for ExitSignal {
            fn drop(&mut self) {
                let _ = self.0.send(());
            }
        }
        thread_local! {
            static ON_EXIT: std::cell::RefCell<Option<ExitSignal>> = const { std::cell::RefCell::new(None) };
        }
        let name = "tidb-lane-retire-test";
        let (exit_tx, exit_rx) = std::sync::mpsc::sync_channel(1);
        spawn_lane_detached_with_keep_alive(
            name,
            std::time::Duration::from_millis(10),
            Box::new(move || {
                ON_EXIT.with(|signal| *signal.borrow_mut() = Some(ExitSignal(exit_tx)))
            }),
        );
        exit_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the idle lane thread exited");
        assert!(idle_lanes()
            .lock()
            .unwrap()
            .get(name)
            .is_none_or(Vec::is_empty));
    }

    #[test]
    fn blocking_lane_drop_waits_for_in_flight_work() {
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let pool = LanePool::new("tidb-worker-pool-close-test", 1);
        pool.submit(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        })
        .unwrap();
        started_rx.recv().unwrap();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            release_tx.send(()).unwrap();
        });
        drop(pool);
        releaser.join().unwrap();
    }

    /// The pool wakes a worker only when one is parked, so a lost wakeup
    /// would strand a task forever. Enqueue from several threads, both while
    /// the pool is idle and while it is saturated, and require every task to
    /// run.
    #[test]
    fn every_task_runs_whether_the_pool_is_idle_or_busy() {
        const PRODUCERS: usize = 4;
        const PER_PRODUCER: usize = 64;
        let (tx, rx) = std::sync::mpsc::channel::<usize>();
        // A first wave while the pool is idle (every worker parked).
        for index in 0..PER_PRODUCER {
            let tx = tx.clone();
            enqueue(Box::new(move || {
                let _ = tx.send(index);
            }));
        }
        // Then waves from several threads at once, on a pool that is still
        // draining the first.
        let threads: Vec<_> = (0..PRODUCERS)
            .map(|producer| {
                let tx = tx.clone();
                std::thread::spawn(move || {
                    for index in 0..PER_PRODUCER {
                        let tx = tx.clone();
                        enqueue(Box::new(move || {
                            let _ = tx.send(producer * PER_PRODUCER + index);
                        }));
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().expect("producer");
        }
        drop(tx);
        let mut seen = 0;
        while rx.recv_timeout(std::time::Duration::from_secs(30)).is_ok() {
            seen += 1;
        }
        assert_eq!(seen, PER_PRODUCER * (PRODUCERS + 1), "every task ran");
    }

    #[test]
    fn map_preserves_submission_order() {
        let inputs: Vec<usize> = (0..32).collect();
        let out = map(
            inputs.iter().map(|&i| {
                move || {
                    // Variable "work" so completion order differs from submission.
                    if i % 3 == 0 {
                        std::thread::sleep(std::time::Duration::from_micros(50));
                    }
                    i * i
                }
            }),
            4,
        );
        assert_eq!(out, inputs.iter().map(|&i| i * i).collect::<Vec<_>>());
    }

    #[test]
    fn shared_arc_state_reaches_the_worker() {
        let data = Arc::new(vec![1u64, 2, 3]);
        let total = map(data.iter().map(|&v| move || v * 10).collect::<Vec<_>>(), 2);
        assert_eq!(total, vec![10, 20, 30]);
    }

    #[test]
    fn owned_buffers_round_trip_through_tasks() {
        let out = map((0..8).map(|i| move || vec![i; 3]), 4);
        assert_eq!(out.len(), 8);
        assert_eq!(out[7], vec![7, 7, 7]);
    }
}
