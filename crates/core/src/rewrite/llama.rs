//! The built-in rewriter: a llama.cpp server the app starts itself, running Gemma 4 E2B with its
//! audio encoder, so the model reads the transcript and hears the recording.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::{OpenAiCompatible, RewriteRequest, Rewriter};
use crate::{Error, Result};

/// The model files and the server binary of the built-in rewriter.
#[derive(Debug, Clone)]
pub struct LlamaServer {
    /// `llama-server` from a llama.cpp release.
    pub binary: PathBuf,
    /// The language model, for example `gemma-4-E2B-it-Q4_K_M.gguf`.
    pub model: PathBuf,
    /// The multimodal projector, for example `mmproj-gemma-4-E2B-it-Q8_0.gguf`. Without it the
    /// model reads the transcript only.
    pub mmproj: Option<PathBuf>,
    /// Where the server writes its output. Empty discards it.
    pub log: Option<PathBuf>,
}

/// A running `llama-server` on a free local port.
///
/// The server runs under a small shell that watches a pipe from this process. When this process
/// exits, even by a crash, the pipe closes and the shell stops the server, so no model stays
/// loaded in the background.
struct Running {
    shell: Child,
    stdin: Option<ChildStdin>,
    base_url: String,
}

impl LlamaServer {
    fn start(&self) -> Result<Running> {
        for (what, path) in [
            ("llama-server", &self.binary),
            ("rewrite model", &self.model),
        ]
        .into_iter()
        .chain(self.mmproj.as_ref().map(|p| ("audio projector", p)))
        {
            if !path.exists() {
                return Err(Error::Rewrite(format!(
                    "{what} not found at {}",
                    path.display()
                )));
            }
        }
        let port = TcpListener::bind("127.0.0.1:0")
            .and_then(|l| l.local_addr())
            .map_err(|e| Error::Rewrite(format!("no free port for llama-server: {e}")))?
            .port();
        let mut args: Vec<String> = vec![
            "-m".into(),
            path_arg(&self.model),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            port.to_string(),
            // One dictation at a time, and a context that fits the prompt, the examples, and a
            // long utterance's audio.
            "-np".into(),
            "1".into(),
            "-c".into(),
            "8192".into(),
            "-ngl".into(),
            "99".into(),
            "--jinja".into(),
            // Gemma 4 thinks before half its answers unless told not to, which costs seconds and
            // made its rewrites worse.
            "--chat-template-kwargs".into(),
            r#"{"enable_thinking":false}"#.into(),
        ];
        if let Some(mmproj) = &self.mmproj {
            args.extend(["--mmproj".into(), path_arg(mmproj)]);
        }
        let (stdout, stderr) = match &self.log {
            Some(path) => {
                let file = File::create(path)
                    .map_err(|e| Error::Rewrite(format!("{}: {e}", path.display())))?;
                let copy = file
                    .try_clone()
                    .map_err(|e| Error::Rewrite(e.to_string()))?;
                (Stdio::from(file), Stdio::from(copy))
            }
            None => (Stdio::null(), Stdio::null()),
        };
        // The shell keeps its stdin on fd 3 (a background job's stdin is /dev/null), starts the
        // server, and has a background reader stop it when the pipe closes. It exits with the
        // server, so a crashed server shows as an exited child.
        let mut shell = Command::new("/bin/sh")
            .arg("-c")
            .arg(r#"exec 3<&0; "$@" 3<&- & server=$!; (read _ <&3; kill "$server" 2>/dev/null) & wait "$server""#)
            .arg("sh")
            .arg(&self.binary)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(stdout)
            .stderr(stderr)
            .spawn()
            .map_err(|e| Error::Rewrite(format!("starting llama-server: {e}")))?;
        let stdin = shell.stdin.take();
        Ok(Running {
            shell,
            stdin,
            base_url: format!("http://127.0.0.1:{port}"),
        })
    }

    /// The last lines of the server log, to explain a server that never came up.
    fn log_tail(&self) -> String {
        let Some(file) = self.log.as_ref().and_then(|p| File::open(p).ok()) else {
            return String::new();
        };
        let lines: Vec<String> = BufReader::new(file).lines().map_while(|l| l.ok()).collect();
        let tail = lines[lines.len().saturating_sub(5)..].join("\n");
        if tail.is_empty() {
            tail
        } else {
            format!(":\n{tail}")
        }
    }
}

fn path_arg(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

impl Running {
    fn exited(&mut self) -> bool {
        matches!(self.shell.try_wait(), Ok(Some(_)))
    }

    fn healthy(&self) -> bool {
        ureq::get(&format!("{}/health", self.base_url))
            .config()
            .timeout_global(Some(Duration::from_secs(2)))
            .build()
            .call()
            .is_ok()
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        // Closing the pipe tells the shell to stop the server.
        drop(self.stdin.take());
        let _ = self.shell.wait();
    }
}

/// The built-in rewriter. The server starts on the first rewrite, or earlier with
/// [`Builtin::start`], and loads the model in the background; the first rewrite waits for it. A
/// server that stopped starts again on the next rewrite.
pub struct Builtin {
    server: LlamaServer,
    running: Mutex<Option<Running>>,
    ready: Mutex<bool>,
    /// How long the first rewrite waits for the model to load.
    pub load_timeout: Duration,
    /// How long one rewrite may take.
    pub timeout: Duration,
}

impl Builtin {
    pub fn new(server: LlamaServer) -> Self {
        Self {
            server,
            running: Mutex::new(None),
            ready: Mutex::new(false),
            load_timeout: Duration::from_secs(120),
            timeout: Duration::from_secs(20),
        }
    }

    /// Starts the server now, so the model loads before the first rewrite needs it.
    pub fn start(&self) -> Result<()> {
        let mut running = self.running.lock().unwrap();
        if running.is_none() {
            *running = Some(self.server.start()?);
            *self.ready.lock().unwrap() = false;
        }
        Ok(())
    }

    /// The server's address once it answers, restarting it if it stopped.
    fn base_url(&self) -> Result<String> {
        let mut running = self.running.lock().unwrap();
        let mut ready = self.ready.lock().unwrap();
        if running.as_mut().is_none_or(|r| r.exited()) {
            *running = Some(self.server.start()?);
            *ready = false;
        }
        let server = running.as_mut().unwrap();
        if !*ready {
            let deadline = Instant::now() + self.load_timeout;
            while !server.healthy() {
                if server.exited() || Instant::now() > deadline {
                    let tail = self.server.log_tail();
                    *running = None;
                    return Err(Error::Rewrite(format!("llama-server did not start{tail}")));
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            *ready = true;
        }
        Ok(server.base_url.clone())
    }
}

impl Rewriter for Builtin {
    fn rewrite(&self, request: &RewriteRequest) -> Result<Option<String>> {
        OpenAiCompatible {
            base_url: format!("{}/v1", self.base_url()?),
            api_key: None,
            model: "builtin".into(),
            timeout: self.timeout,
            send_audio: self.server.mmproj.is_some(),
        }
        .rewrite(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_missing_files_before_starting() {
        let server = LlamaServer {
            binary: "/bin/sh".into(),
            model: "/nonexistent/model.gguf".into(),
            mmproj: None,
            log: None,
        };
        let error = Builtin::new(server).start().err().unwrap().to_string();
        assert!(error.contains("rewrite model not found"), "{error}");
    }

    #[test]
    fn stops_the_server_when_dropped() {
        // A stand-in server that only sleeps: dropping the handle must end it.
        let dir = std::env::temp_dir().join(format!("md-llama-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fake = dir.join("llama-server");
        std::fs::write(
            &fake,
            "#!/bin/sh\necho $$ > \"$(dirname \"$0\")/pid\"\nexec sleep 30\n",
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
        let model = dir.join("model.gguf");
        std::fs::write(&model, "").unwrap();
        let server = LlamaServer {
            binary: fake,
            model,
            mmproj: None,
            log: None,
        };
        let running = server.start().unwrap();
        let pid_file = dir.join("pid");
        let deadline = Instant::now() + Duration::from_secs(5);
        while !pid_file.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        let pid = std::fs::read_to_string(&pid_file)
            .unwrap()
            .trim()
            .to_owned();
        let alive = || {
            Command::new("kill")
                .args(["-0", &pid])
                .status()
                .unwrap()
                .success()
        };
        assert!(alive());
        drop(running);
        assert!(!alive());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
