
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use crate::host::generate::{self, Delta, GenerationOutput, GenerationRequest};
use crate::host::load;
use crate::host::runtime::Workspace;
use crate::host::tokenizer::Tokenizer;
use furiosa_opt_std::prelude::*;

pub enum StreamEvent {
    Delta(String),
    ReasoningDelta(String),
    Done(GenerationOutput),
    Error(String),
}

pub struct Job {
    pub request: GenerationRequest,
    pub stream: bool,
    pub events: mpsc::Sender<StreamEvent>,
}

#[derive(Clone)]
pub struct WorkerHandle {
    jobs: mpsc::Sender<Job>,
    alive: Arc<AtomicBool>,
}

impl WorkerHandle {
    pub fn submit(&self, job: Job) {
        if let Err(mpsc::SendError(job)) = self.jobs.send(job) {
            let _ = job
                .events
                .send(StreamEvent::Error("generation worker is not running".to_owned()));
        }
    }

    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }
}

pub fn spawn(tokenizer: Tokenizer) -> Result<WorkerHandle, String> {
    let (jobs_tx, jobs_rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    let alive = Arc::new(AtomicBool::new(true));
    let worker_alive = Arc::clone(&alive);
    std::thread::Builder::new()
        .name("gemma4-worker".into())
        .spawn(move || {
            let _alive = AliveFlag(worker_alive);
            run(tokenizer, jobs_rx, ready_tx);
        })
        .map_err(|err| format!("failed to spawn the gemma4 generation worker thread: {err}"))?;
    match ready_rx.recv() {
        Ok(Ok(())) => Ok(WorkerHandle { jobs: jobs_tx, alive }),
        Ok(Err(message)) => Err(message),
        Err(_) => Err("generation worker thread exited before finishing startup".to_owned()),
    }
}

struct AliveFlag(Arc<AtomicBool>);

impl Drop for AliveFlag {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

fn run(tokenizer: Tokenizer, jobs: mpsc::Receiver<Job>, ready: mpsc::Sender<Result<(), String>>) {
    let rt = match tokio::runtime::Builder::new_current_thread().build() {
        Ok(rt) => rt,
        Err(err) => {
            let _ = ready.send(Err(format!(
                "failed to build the gemma4 generation worker's executor: {err}"
            )));
            return;
        }
    };

    let setup = rt.block_on(async {
        let mut ctx = Context::acquire();
        let load_start = Instant::now();
        let model = load::load_model(&mut ctx).await.map_err(|err| err.to_string())?;
        eprintln!("model loaded in {:.2?}", load_start.elapsed());
        let workspace = Workspace::new(&mut ctx, &model).await;
        Ok::<_, String>((ctx, model, workspace))
    });
    let (mut ctx, model, mut workspace) = match setup {
        Ok(value) => value,
        Err(message) => {
            let _ = ready.send(Err(message));
            return;
        }
    };
    let _ = ready.send(Ok(()));

    for job in jobs {
        let Job {
            request,
            stream,
            events,
        } = job;
        let delta_events = events.clone();
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rt.block_on(generate::generate(
                &mut ctx,
                &model,
                &tokenizer,
                &mut workspace,
                request,
                |delta| {
                    if !stream {
                        return true;
                    }
                    let event = match delta {
                        Delta::Content(text) => StreamEvent::Delta(text.to_owned()),
                        Delta::Reasoning(text) => StreamEvent::ReasoningDelta(text.to_owned()),
                    };
                    delta_events.send(event).is_ok()
                },
            ))
        }));
        let outcome = match panicked {
            Ok(Ok(output)) => StreamEvent::Done(output),
            Ok(Err(err)) => StreamEvent::Error(err.to_string()),
            Err(panic) => {
                let message = panic_message(&panic);
                eprintln!("generation worker panicked, taking the worker offline: {message}");
                let _ = events.send(StreamEvent::Error(format!(
                    "internal error: generation panicked ({message})"
                )));
                return;
            }
        };
        let _ = events.send(outcome);
    }
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_owned())
}
