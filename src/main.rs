//! `decryptd` — a generic volunteer GPU job runner for decrypt.
//!
//! decryptd knows nothing about bloom filters, RNG, or BIP39. It just:
//!   1. asks the coordinator for a chunk,
//!   2. downloads the cubin(s) (job.zip) and the opaque data blob (data.xz),
//!   3. loads the cubin for the local GPU and launches its kernel over the range
//!      (the kernel does all the real work and writes output records),
//!   4. gathers the output records, compresses them (xz), and uploads.
//!
//! All job-specific logic lives in the cubin + the data blob, produced by the
//! coordinator. See docs/cluster-protocol.md and the kernel ABI in `cuda.rs`.

mod cuda;
mod submit;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use serde::Deserialize;

#[derive(Parser)]
#[command(name = "decryptd", about = "Generic GPU job runner for decrypt")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run as a worker: pull GPU chunks, run the kernel, upload results.
    Run(RunArgs),
    /// Publish a job to the KLB Decrypt/* API: upload engine.zip + data.bin.xz as
    /// Decrypt\Data blobs and create a Decrypt\Job over a range referencing both.
    Submit(submit::SubmitArgs),
}

#[derive(Parser)]
struct RunArgs {
    /// Coordinator base URL, e.g. <https://sweep.example.com>
    #[arg(long, env = "DECRYPTD_SERVER")]
    server: String,
    /// Working directory for the worker id, artifact cache, and scratch.
    #[arg(long, env = "DECRYPTD_WORKDIR", default_value = "decryptd-data")]
    workdir: PathBuf,
    /// Optional bearer token for the coordinator.
    #[arg(long, env = "DECRYPTD_TOKEN")]
    token: Option<String>,
    /// Run a single chunk then exit (default: loop forever).
    #[arg(long)]
    once: bool,
    /// Seconds to wait when the coordinator has no work (capped backoff base).
    #[arg(long, default_value_t = 15)]
    idle_secs: u64,
}

// ----------------------------------------------------------------- assignment JSON
#[derive(Deserialize)]
struct Assignment {
    assignment_id: String,
    #[allow(dead_code)]
    #[serde(default)]
    job_id: String,
    range: RangeSpec,
    launch: Launch,
    artifacts: Artifacts,
}
#[derive(Deserialize)]
struct RangeSpec {
    start: u64,
    end: u64,
}
/// Generic kernel launch parameters (no job semantics — just how to run the cubin).
#[derive(Deserialize)]
struct Launch {
    /// Kernel entry-point symbol name.
    #[serde(default = "d_entry")]
    entry: String,
    /// Output record size in bytes (e.g. 28 = u64 seed + 20-byte address).
    record_size: u32,
    /// Output buffer capacity in records.
    #[serde(default = "d_out_cap")]
    out_cap: u32,
    /// CUDA block size.
    #[serde(default = "d_block")]
    block: u32,
    /// Work-items per kernel launch (tiles a large range).
    #[serde(default = "d_tile")]
    tile: u64,
}
fn d_entry() -> String {
    "decrypt".into()
}
fn d_out_cap() -> u32 {
    1 << 20
}
fn d_block() -> u32 {
    256
}
fn d_tile() -> u64 {
    1 << 24
}

#[derive(Deserialize)]
struct Artifacts {
    job: Blob,
    #[serde(default)]
    data: Option<Blob>,
}
#[derive(Deserialize)]
struct Blob {
    url: String,
    sha256: String,
}

// --------------------------------------------------------------------------- http
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn sha256_hex(bytes: &[u8]) -> String {
    hex(purecrypto::hash::sha256(bytes).as_ref())
}

fn os_tag() -> &'static str {
    if cfg!(target_os = "windows") {
        "windows-x64"
    } else if cfg!(target_os = "macos") {
        "macos-arm64"
    } else {
        "linux-x64"
    }
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn req(method: &str, url: &str, token: &Option<String>) -> Result<rsurl::Request> {
    let mut r = rsurl::Request::new(method, url).map_err(|e| anyhow!("{e}"))?;
    if let Some(t) = token {
        r = r.header("authorization", &format!("Bearer {t}"));
    }
    Ok(r.max_time(Duration::from_secs(300)))
}

/// Ask for a chunk. `Ok(None)` on HTTP 204 (no work).
fn request_work(args: &RunArgs, worker: &str) -> Result<Option<Assignment>> {
    let (gpu, maj, min) = cuda::probe_device().unwrap_or(("none".into(), 0, 0));
    let url = format!(
        "{}/v1/work?worker={worker}&os={}&cc={maj}{min}&ver={}&gpu={}",
        args.server.trim_end_matches('/'),
        os_tag(),
        env!("CARGO_PKG_VERSION"),
        urlencode(&gpu),
    );
    let resp = req("GET", &url, &args.token)?
        .send()
        .map_err(|e| anyhow!("GET /v1/work: {e}"))?;
    if resp.status == 204 {
        return Ok(None);
    }
    if resp.status >= 400 {
        bail!("GET /v1/work → HTTP {}", resp.status);
    }
    Ok(Some(
        serde_json::from_slice(&resp.body).context("parsing assignment JSON")?,
    ))
}

/// Download (or reuse from cache) an artifact, verifying its SHA-256.
fn fetch_blob(args: &RunArgs, blob: &Blob, cache: &Path) -> Result<Vec<u8>> {
    std::fs::create_dir_all(cache)?;
    let path = cache.join(&blob.sha256);
    if let Ok(bytes) = std::fs::read(&path)
        && sha256_hex(&bytes) == blob.sha256
    {
        return Ok(bytes); // cache hit
    }
    let url = if blob.url.starts_with("http") {
        blob.url.clone()
    } else {
        format!("{}{}", args.server.trim_end_matches('/'), blob.url)
    };
    eprintln!(
        "[decryptd] downloading {} ({})",
        blob.url,
        &blob.sha256[..12.min(blob.sha256.len())]
    );
    let resp = req("GET", &url, &args.token)?
        .send()
        .map_err(|e| anyhow!("GET {url}: {e}"))?;
    if resp.status >= 400 {
        bail!("GET {url} → HTTP {}", resp.status);
    }
    let got = sha256_hex(&resp.body);
    if got != blob.sha256 {
        bail!("sha256 mismatch for {url}: got {got}, want {}", blob.sha256);
    }
    std::fs::write(&path, &resp.body)?;
    Ok(resp.body)
}

/// Extract job.zip and return every `*.sm<NN>.cubin`'s bytes, highest-arch first.
fn ensure_cubins(args: &RunArgs, blob: &Blob) -> Result<Vec<Vec<u8>>> {
    let zip_bytes = fetch_blob(args, blob, &args.workdir.join("cache"))?;
    let dest = args.workdir.join("jobs").join(&blob.sha256);
    if !dest.join(".unpacked").exists() {
        std::fs::create_dir_all(&dest)?;
        let mut zip =
            zip::ZipArchive::new(std::io::Cursor::new(&zip_bytes)).context("opening job.zip")?;
        zip.extract(&dest).context("extracting job.zip")?;
        std::fs::write(dest.join(".unpacked"), b"")?;
    }
    // Collect cubins by arch parsed from `.sm<NN>.cubin`, then sort high→low.
    let mut found: Vec<(u32, PathBuf)> = Vec::new();
    for e in std::fs::read_dir(&dest)?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if let Some(i) = name.find(".sm")
            && let Some(rest) = name[i + 3..].strip_suffix(".cubin")
            && let Ok(arch) = rest.parse::<u32>()
        {
            found.push((arch, e.path()));
        }
    }
    if found.is_empty() {
        bail!("job.zip contains no *.sm<NN>.cubin files");
    }
    found.sort_by_key(|b| std::cmp::Reverse(b.0));
    Ok(found
        .iter()
        .filter_map(|(_, p)| std::fs::read(p).ok())
        .collect())
}

/// Download + xz-decompress the opaque data blob (cached per content hash).
fn ensure_data(args: &RunArgs, blob: &Blob) -> Result<Vec<u8>> {
    let raw = args
        .workdir
        .join("data")
        .join(format!("{}.bin", blob.sha256));
    if let Ok(bytes) = std::fs::read(&raw) {
        return Ok(bytes);
    }
    let xz = fetch_blob(args, blob, &args.workdir.join("cache"))?;
    let data = compcol::vec::decompress_to_vec::<compcol::xz::Xz>(&xz)
        .map_err(|e| anyhow!("xz-decompressing data: {e:?}"))?;
    std::fs::create_dir_all(raw.parent().unwrap())?;
    std::fs::write(&raw, &data)?;
    Ok(data)
}

// --------------------------------------------------------------------------- run
fn run_once(args: &RunArgs, worker: &str) -> Result<bool> {
    let Some(a) = request_work(args, worker)? else {
        return Ok(false);
    };
    let cubins = ensure_cubins(args, &a.artifacts.job)?;
    let data = match &a.artifacts.data {
        Some(b) => ensure_data(args, b)?,
        None => Vec::new(),
    };

    let gpu = cuda::Gpu::load_first(&cubins).map_err(|e| anyhow!(e))?;
    let (maj, min) = gpu.compute_capability();
    let total = a.range.end.saturating_sub(a.range.start).saturating_add(1);
    eprintln!(
        "[decryptd] chunk {} on {} (sm_{maj}{min}): entry={} range {}..={} ({total} items)",
        a.assignment_id,
        gpu.device_name(),
        a.launch.entry,
        a.range.start,
        a.range.end
    );

    let t0 = Instant::now();
    let out = cuda::run_job(
        &gpu,
        &a.launch.entry,
        &data,
        a.range.start,
        a.range.end,
        a.launch.record_size,
        a.launch.out_cap,
        a.launch.block,
        a.launch.tile,
        |done, total| {
            eprint!(
                "\r  {done}/{total} ({:.1}%)   ",
                100.0 * done as f64 / total as f64
            )
        },
    )
    .map_err(|e| anyhow!("run_job: {e}"))?;
    eprintln!();
    let records = out.len() / a.launch.record_size.max(1) as usize;

    // Compress the raw output records and upload.
    let packed = compcol::vec::compress_to_vec::<compcol::xz::Xz>(&out)
        .map_err(|e| anyhow!("xz-compressing result: {e:?}"))?;
    upload_result(
        args,
        &a,
        &packed,
        worker,
        records as u64,
        t0.elapsed().as_secs_f64(),
    )?;
    Ok(true)
}

fn upload_result(
    args: &RunArgs,
    a: &Assignment,
    body: &[u8],
    worker: &str,
    records: u64,
    secs: f64,
) -> Result<()> {
    let url = format!(
        "{}/v1/work/{}/result",
        args.server.trim_end_matches('/'),
        a.assignment_id
    );
    let resp = req("POST", &url, &args.token)?
        .header("content-type", "application/x-xz")
        .header("x-decryptd-records", &records.to_string())
        .header("x-decryptd-record-size", &a.launch.record_size.to_string())
        .header("x-decryptd-seconds", &format!("{secs:.1}"))
        .header("x-decryptd-worker", worker)
        .body(body.to_vec())
        .send()
        .map_err(|e| anyhow!("POST result: {e}"))?;
    if resp.status >= 400 {
        bail!("POST result → HTTP {} {}", resp.status, resp.reason);
    }
    eprintln!(
        "[decryptd] uploaded {records} record(s) ({} B xz) for {} in {secs:.1}s",
        body.len(),
        a.assignment_id
    );
    Ok(())
}

/// Stable per-install worker id, persisted in the work dir.
fn worker_id(workdir: &Path) -> Result<String> {
    let path = workdir.join("worker-id");
    if let Ok(s) = std::fs::read_to_string(&path) {
        let s = s.trim().to_string();
        if !s.is_empty() {
            return Ok(s);
        }
    }
    let host = std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_default();
    let seed = format!(
        "{host}|{}|{:?}",
        std::process::id(),
        std::env::current_exe().ok()
    );
    let id = format!("w-{}", &sha256_hex(seed.as_bytes())[..16]);
    std::fs::create_dir_all(workdir)?;
    std::fs::write(&path, &id)?;
    Ok(id)
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Run(args) => run_worker(args),
        Cmd::Submit(args) => submit::run(args),
    }
}

fn run_worker(args: RunArgs) -> Result<()> {
    std::fs::create_dir_all(&args.workdir)?;
    let worker = worker_id(&args.workdir)?;
    eprintln!(
        "[decryptd] id={worker} os={} server={}",
        os_tag(),
        args.server
    );

    let mut idle = 0u32;
    loop {
        match run_once(&args, &worker) {
            Ok(true) => idle = 0,
            Ok(false) => {
                if args.once {
                    eprintln!("[decryptd] no work; exiting (--once)");
                    return Ok(());
                }
                let wait = args.idle_secs * (1 << idle.min(4));
                eprintln!("[decryptd] no work; sleeping {wait}s");
                std::thread::sleep(Duration::from_secs(wait));
                idle = idle.saturating_add(1);
            }
            Err(e) => {
                eprintln!("[decryptd] error: {e:#}");
                if args.once {
                    return Err(e);
                }
                std::thread::sleep(Duration::from_secs(args.idle_secs));
            }
        }
        if args.once {
            return Ok(());
        }
    }
}
