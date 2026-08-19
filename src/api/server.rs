
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use serde::Serialize;
use serde::de::DeserializeOwned;

use super::handlers;
use super::worker::WorkerHandle;
use crate::host::tokenizer::Tokenizer;

const MAX_BODY_BYTES: u64 = 32 * 1024 * 1024;
const MAX_LINE_BYTES: u64 = 8 * 1024;
const MAX_HEADER_LINES: usize = 100;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_CONNECTIONS: usize = 256;
const IO_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub struct AppState {
    pub worker: WorkerHandle,
    pub tokenizer: Tokenizer,
    pub api_key: Option<String>,
}

pub struct HttpRequest {
    pub method: String,
    pub path: String,
    headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpRequest {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

pub enum ResponseBody {
    Fixed(Vec<u8>),
    Stream(mpsc::Receiver<Vec<u8>>),
}

pub struct HttpResponse {
    pub status: u16,
    pub reason: &'static str,
    pub headers: Vec<(&'static str, String)>,
    pub body: ResponseBody,
}

impl HttpResponse {
    pub fn json<T: Serialize>(status: u16, reason: &'static str, body: &T) -> Self {
        let bytes = serde_json::to_vec(body).expect("response schema always serializes");
        Self {
            status,
            reason,
            headers: vec![("content-type", "application/json".to_owned())],
            body: ResponseBody::Fixed(bytes),
        }
    }

    pub fn event_stream(rx: mpsc::Receiver<Vec<u8>>) -> Self {
        Self {
            status: 200,
            reason: "OK",
            headers: vec![
                ("content-type", "text/event-stream".to_owned()),
                ("cache-control", "no-cache".to_owned()),
            ],
            body: ResponseBody::Stream(rx),
        }
    }
}

pub fn read_json_body<T: DeserializeOwned>(req: &HttpRequest) -> Result<T, HttpResponse> {
    serde_json::from_slice(&req.body)
        .map_err(|err| handlers::error(400, "invalid_request_error", format!("invalid JSON body: {err}")))
}

pub fn serve(addr: SocketAddr, state: AppState) -> io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    eprintln!("gemma4 API server listening on http://{addr}");
    let connections = Arc::new(AtomicUsize::new(0));
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(stream) => stream,
            Err(err) => {
                eprintln!("accept error: {err}");
                continue;
            }
        };
        if connections.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
            connections.fetch_sub(1, Ordering::SeqCst);
            continue;
        }
        let slot = ConnectionSlot(Arc::clone(&connections));
        let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
        let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
        let _ = stream.set_nodelay(true);
        let state = state.clone();
        let spawned = std::thread::Builder::new()
            .name("gemma4-connection".to_owned())
            .spawn(move || {
                let _slot = slot;
                handle_connection(stream, &state);
            });
        if let Err(err) = spawned {
            eprintln!("could not spawn a connection thread: {err}");
        }
    }
    Ok(())
}

struct ConnectionSlot(Arc<AtomicUsize>);

impl Drop for ConnectionSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

fn handle_connection(stream: TcpStream, state: &AppState) {
    let request = {
        let mut reader = BufReader::new(&stream);
        match read_request(&mut reader) {
            Ok(Some(request)) => request,
            Ok(None) => return,
            Err(response) => {
                let _ = write_response(&stream, response);
                return;
            }
        }
    };
    let response = route(request, state);
    if let Err(err) = write_response(&stream, response) {
        eprintln!("error writing response: {err}");
    }
}

fn route(req: HttpRequest, state: &AppState) -> HttpResponse {
    if let Some(unauthorized) = handlers::check_auth(&req, state) {
        return unauthorized;
    }
    let path = req.path.split('?').next().unwrap_or(&req.path);
    match (req.method.as_str(), path) {
        ("GET", "/v1/models") => handlers::list_models(state),
        ("POST", "/v1/chat/completions") => handlers::chat_completions(&req, state),
        ("POST", "/v1/completions") => handlers::completions(&req, state),
        ("POST", "/v1/audio/speech") => handlers::audio_speech(&req),
        ("POST", "/v1/audio/transcriptions") => handlers::audio_transcriptions(&req),
        _ => handlers::not_found(),
    }
}

fn read_request(reader: &mut BufReader<&TcpStream>) -> Result<Option<HttpRequest>, HttpResponse> {
    let request_line = match read_line(reader)? {
        Some(line) => line,
        None => return Ok(None),
    };
    let mut parts = request_line.split(' ');
    let method = parts.next().unwrap_or_default().to_owned();
    let path = parts.next().unwrap_or_default().to_owned();
    if method.is_empty() || path.is_empty() {
        return Err(handlers::error(400, "invalid_request_error", "malformed request line"));
    }

    let mut headers = Vec::new();
    let mut header_bytes = 0usize;
    loop {
        let line = read_line(reader)?
            .ok_or_else(|| handlers::error(400, "invalid_request_error", "connection closed while reading headers"))?;
        if line.is_empty() {
            break;
        }
        header_bytes += line.len();
        if header_bytes > MAX_HEADER_BYTES || headers.len() >= MAX_HEADER_LINES {
            return Err(handlers::error(
                431,
                "invalid_request_error",
                "request headers too large",
            ));
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_owned(), value.trim().to_owned()));
        }
    }

    if headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("transfer-encoding"))
    {
        return Err(handlers::error(
            501,
            "invalid_request_error",
            "transfer-encoding is not supported; send a content-length body",
        ));
    }

    let mut content_length = 0u64;
    let mut seen_length = false;
    for (_, value) in headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
    {
        let parsed = value.trim().parse::<u64>().map_err(|_| {
            handlers::error(
                400,
                "invalid_request_error",
                format!("malformed content-length header: {value}"),
            )
        })?;
        if seen_length && parsed != content_length {
            return Err(handlers::error(
                400,
                "invalid_request_error",
                "conflicting content-length headers",
            ));
        }
        content_length = parsed;
        seen_length = true;
    }
    if content_length > MAX_BODY_BYTES {
        return Err(handlers::error(413, "invalid_request_error", "request body too large"));
    }
    let mut body = Vec::new();
    let read = reader.take(content_length).read_to_end(&mut body).map_err(|err| {
        handlers::error(
            400,
            "invalid_request_error",
            format!("could not read request body: {err}"),
        )
    })?;
    if read as u64 != content_length {
        return Err(handlers::error(
            400,
            "invalid_request_error",
            "request body is shorter than its content-length",
        ));
    }

    Ok(Some(HttpRequest {
        method,
        path,
        headers,
        body,
    }))
}

fn read_line(reader: &mut BufReader<&TcpStream>) -> Result<Option<String>, HttpResponse> {
    let mut raw = Vec::new();
    let read = reader
        .take(MAX_LINE_BYTES)
        .read_until(b'\n', &mut raw)
        .map_err(|err| handlers::error(400, "invalid_request_error", format!("could not read request: {err}")))?;
    if read == 0 {
        return Ok(None);
    }
    if !raw.ends_with(b"\n") {
        return Err(handlers::error(
            431,
            "invalid_request_error",
            "request line or header too large",
        ));
    }
    while raw.last() == Some(&b'\n') || raw.last() == Some(&b'\r') {
        raw.pop();
    }
    String::from_utf8(raw)
        .map(Some)
        .map_err(|_| handlers::error(400, "invalid_request_error", "request line is not valid UTF-8"))
}

fn write_response(stream: &TcpStream, response: HttpResponse) -> io::Result<()> {
    let mut writer = stream;
    let mut head = format!("HTTP/1.1 {} {}\r\n", response.status, response.reason);
    for (name, value) in &response.headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    match response.body {
        ResponseBody::Fixed(bytes) => {
            head.push_str(&format!("content-length: {}\r\n", bytes.len()));
            head.push_str("connection: close\r\n\r\n");
            writer.write_all(head.as_bytes())?;
            writer.write_all(&bytes)?;
        }
        ResponseBody::Stream(rx) => {
            head.push_str("connection: close\r\n\r\n");
            writer.write_all(head.as_bytes())?;
            for chunk in rx {
                writer.write_all(&chunk)?;
                writer.flush()?;
            }
        }
    }
    Ok(())
}
