/// Errors of the dictation core. Platform shells add their own context when they show one.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("model: {0}")]
    Model(String),
    #[error("context provider: {0}")]
    Context(String),
    #[error("rewrite: {0}")]
    Rewrite(String),
    #[error("text sink: {0}")]
    Sink(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
