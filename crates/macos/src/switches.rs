//! Menu switches for the rewrite and the screen read. The engine holds these wrappers, and the
//! menu flips their flags while it runs.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use mini_dictator_core::context::{ContextProvider, ScreenContext};
use mini_dictator_core::rewrite::{RewriteRequest, Rewriter};
use mini_dictator_core::Result;

pub struct Switchable<T> {
    pub inner: T,
    pub enabled: Arc<AtomicBool>,
}

impl<T> Switchable<T> {
    pub fn new(inner: T, enabled: bool) -> Self {
        Self {
            inner,
            enabled: Arc::new(AtomicBool::new(enabled)),
        }
    }

    fn on(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

impl<T: ContextProvider> ContextProvider for Switchable<T> {
    fn snapshot(&self) -> Result<ScreenContext> {
        if self.on() {
            self.inner.snapshot()
        } else {
            Ok(ScreenContext::empty())
        }
    }
}

impl Rewriter for Switchable<Option<Box<dyn Rewriter>>> {
    fn rewrite(&self, request: &RewriteRequest) -> Result<Option<String>> {
        match (&self.inner, self.on()) {
            (Some(rewriter), true) => rewriter.rewrite(request),
            _ => Ok(None),
        }
    }

    fn reads_uncertainty(&self) -> bool {
        self.on() && self.inner.as_ref().is_some_and(|r| r.reads_uncertainty())
    }
}
