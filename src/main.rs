// In the Windows GUI build the worker lives in the tray, so don't attach a
// console window (which would otherwise flash up on launch). Console/headless
// builds keep the default subsystem so stderr logging stays visible.
#![cfg_attr(
    all(feature = "gui", target_os = "windows"),
    windows_subsystem = "windows"
)]

//! `decryptd` — a GPU job runner for decrypt.
//!
//! decryptd knows nothing about bloom filters, RNG, or BIP39. It just:
//!   1. claims a fragment of work from the platform (`Decrypt/Job:pullOne`),
//!   2. downloads the job's blobs (`engine.zip` + an optional compressed `data`
//!      blob — xz or gzip, auto-detected) from the inline URLs the pull
//!      response carries,
//!   3. reads launch parameters from `manifest.json` inside `engine.zip`, loads the
//!      cubin for the local GPU, and launches its kernel over the fragment range
//!      (the kernel does all the real work and writes output records),
//!   4. gathers the output records, compresses them (xz), and submits them back with
//!      `Decrypt/Job:submit`.
//!
//! These stages are pipelined: the next fragment downloads while the GPU runs the
//! current one, and finished results upload while the GPU runs the next. Each GPU
//! gets its own independent pipeline (prefetch → run → upload), and `--jobs` sets
//! how many fragments run concurrently *on each* GPU; a shared in-flight set and
//! blob-download map keep the GPUs from duplicating each other's work.
//!
//! `pullOne`/`submit` and the `Decrypt/Data` blobs are anonymous-accessible, so a
//! worker needs no credentials. All job-specific logic lives in the cubin + the data
//! blob, produced by the job's publisher. See the kernel ABI in `cuda.rs`.

mod cuda;
#[cfg(all(feature = "gui", any(target_os = "linux", target_os = "windows")))]
mod gui;
#[cfg(all(feature = "gui", any(target_os = "linux", target_os = "windows")))]
mod nvml;

use std::collections::{HashMap, VecDeque};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use klbfw::{Config, RestContext};
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

/// Run as a worker: claim Decrypt/Job fragments, run the kernel, submit results.
#[derive(Parser)]
#[command(name = "decryptd", about = "Generic GPU job runner for decrypt")]
struct RunArgs {
    /// KarpelesLab platform host (pullOne/submitFragment are anonymous — no key needed).
    #[arg(long, env = "DECRYPTD_HOST", default_value = "www.atonline.com")]
    host: String,
    /// Working directory for the blob cache, worker id, and scratch. Defaults to a
    /// per-user data dir for the GUI build (it may be launched from a non-writable
    /// CWD like System32) and to `./decryptd-data` for the console build.
    #[arg(long, env = "DECRYPTD_WORKDIR")]
    workdir: Option<PathBuf>,
    /// Claim a single fragment then exit (default: loop forever).
    #[arg(long)]
    once: bool,
    /// Seconds to wait before retrying when no work is available.
    #[arg(long, default_value_t = 60)]
    idle_secs: u64,
    /// Number of GPU jobs to run concurrently **per GPU** (default: one at a time).
    /// Download of the next job and upload of finished results always overlap the
    /// GPU run. With N GPUs in use this is N×jobs concurrent runs in total.
    #[arg(long, default_value_t = 1)]
    jobs: usize,
    /// Comma-separated GPU ordinals to use, e.g. "0,2". Default: every detected
    /// GPU. Ordinals are post-`CUDA_VISIBLE_DEVICES`. Each gets its own work queue.
    #[arg(long)]
    gpus: Option<String>,
    /// Maximum on-disk blob cache size, in GB. Once exceeded, the oldest cached
    /// blobs are evicted. Eviction is safe: a running job holds its blob in memory,
    /// so the only cost of dropping a file is a re-download on a later cache miss.
    #[arg(long, default_value_t = 20)]
    cache_max_gb: u64,
}

impl RunArgs {
    /// The resolved working directory: the `--workdir`/`DECRYPTD_WORKDIR` value if
    /// given, else [`default_workdir`].
    fn workdir(&self) -> PathBuf {
        self.workdir.clone().unwrap_or_else(default_workdir)
    }
}

/// Default working directory. The GUI build can be started from any CWD (an
/// Explorer double-click, a login item, a service), and a CWD-relative folder may
/// not be creatable there — a silent failure that strands the tray on "Waiting".
/// So the GUI build defaults to a stable per-user data dir; the console build,
/// run from an operator-chosen directory, keeps `./decryptd-data`.
fn default_workdir() -> PathBuf {
    #[cfg(feature = "gui")]
    {
        #[cfg(windows)]
        if let Some(base) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(base).join("decryptd");
        }
        #[cfg(not(windows))]
        {
            if let Some(base) = std::env::var_os("XDG_DATA_HOME") {
                return PathBuf::from(base).join("decryptd");
            }
            if let Some(home) = std::env::var_os("HOME") {
                return PathBuf::from(home).join(".local/share/decryptd");
            }
        }
    }
    PathBuf::from("decryptd-data")
}

// ------------------------------------------------------------- pullOne response
/// `Decrypt/Job:pullOne` payload — a claimed fragment, its parent job's blobs, and
/// the single-use key that authenticates this fragment's result submission.
#[derive(Deserialize)]
struct Pull {
    #[serde(rename = "Fragment")]
    fragment: Fragment,
    #[serde(rename = "Job")]
    job: Job,
    #[serde(rename = "Response_Key")]
    response_key: String,
}
#[derive(Deserialize)]
struct Fragment {
    /// Fragment UUID — used to detect a fragment the platform re-issued to us while
    /// we're still processing it (so we don't run the same work twice).
    #[serde(rename = "Decrypt_Job_Fragment__")]
    id: String,
    #[serde(rename = "Range_Start", deserialize_with = "de_u64")]
    range_start: u64,
    /// Exclusive — the platform's ranges are half-open `[Range_Start, Range_End)`.
    #[serde(rename = "Range_End", deserialize_with = "de_u64")]
    range_end: u64,
}
#[derive(Deserialize)]
struct Job {
    #[serde(rename = "Data", default)]
    data: Vec<DataRef>,
}
#[derive(Deserialize)]
struct DataRef {
    /// Inline signed download URL.
    #[serde(rename = "Url")]
    url: Option<String>,
    /// Original filename — `engine.zip` vs the `data` blob tells the two apart.
    #[serde(rename = "Filename", default)]
    filename: String,
    /// Content SHA-256, used to verify + cache the download.
    #[serde(rename = "Hash", default)]
    hash: String,
}

/// BIGINT columns serialize as JSON strings; accept a number too, just in case.
fn de_u64<'de, D: serde::Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
    use serde::de::Error;
    match Value::deserialize(d)? {
        Value::String(s) => s.parse().map_err(Error::custom),
        Value::Number(n) => n.as_u64().ok_or_else(|| Error::custom("not a u64")),
        other => Err(Error::custom(format!("expected u64, got {other}"))),
    }
}

// ----------------------------------------------------------------- engine.zip
/// An arch-tagged cubin from `engine.zip`: compute capability `X.Y` encoded as
/// `X*10+Y` (matching the `smNN` filename tag), paired with the raw cubin bytes.
type Cubin = (u32, Vec<u8>);

/// `manifest.json` shipped inside `engine.zip` — the generic kernel launch
/// parameters that the platform's Decrypt/Job row does not carry.
#[derive(Deserialize)]
struct Manifest {
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

// --------------------------------------------------------------------- helpers
fn sha256_hex(bytes: &[u8]) -> String {
    purecrypto::hash::sha256(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Parse a 64-char hex SHA-256 into raw bytes, or `None` if it isn't one.
fn parse_sha256_hex(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 32 * 2 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(hex.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

/// Download a Job.Data blob from its inline URL, caching it under its content hash.
///
/// These blobs can be hundreds of MiB, so the body is *streamed to a file*
/// rather than buffered in memory. rsurl's resumable downloader continues a
/// partial `.part` across retries and process restarts (so a dropped
/// connection resumes instead of restarting) and — given the content hash —
/// verifies the finished file end-to-end.
///
/// Single-stream (not segmented/parallel): segmented mode probes the size with
/// a `HEAD`, but these blob URLs are SigV4-presigned for `GET` only, so a
/// `HEAD` returns 403. A plain `GET` stream still gets the essentials — no
/// size cap, and `Range`-based resume on a drop.
fn fetch_blob(args: &RunArgs, downloads: &Downloads, d: &DataRef) -> Result<Vec<u8>> {
    let url = d
        .url
        .as_deref()
        .ok_or_else(|| anyhow!("Job.Data entry has no Url"))?;

    // The platform inlines small blobs as `data:` URIs instead of a presigned
    // HTTP URL. rsurl only speaks HTTP(S), so decode these ourselves rather than
    // handing an unsupported scheme to the downloader. The payload is already in
    // hand, so there's nothing to cache or resume — just decode and (if the row
    // carries one) verify the content hash.
    if url.starts_with("data:") {
        let bytes = decode_data_url(url)?;
        if !d.hash.is_empty() {
            let got = sha256_hex(&bytes);
            if got != d.hash {
                bail!(
                    "data: URL hash mismatch for {}: {got} != {}",
                    d.filename,
                    d.hash
                );
            }
        }
        return Ok(bytes);
    }

    let cache = args.workdir().join("cache");
    std::fs::create_dir_all(&cache)?;
    // `Hash` is the blob's content SHA-256, so it doubles as a cache key + checksum.
    let cache_path = (!d.hash.is_empty()).then(|| cache.join(&d.hash));
    if let Some(p) = &cache_path
        && let Ok(bytes) = std::fs::read(p)
        && sha256_hex(&bytes) == d.hash
    {
        return Ok(bytes); // cache hit
    }

    // Serialize downloaders of the same content hash across all GPUs' prefetch
    // threads (they share this cache). `hash_lock` is held for the whole
    // download so a peer targeting the same blob waits here rather than racing
    // on the same file; it's declared before `_guard` so the Arc outlives the
    // guard that borrows it.
    let hash_lock = (!d.hash.is_empty()).then(|| {
        downloads
            .lock()
            .unwrap()
            .entry(d.hash.clone())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    });
    let _guard = hash_lock
        .as_ref()
        .map(|l| l.lock().unwrap_or_else(|e| e.into_inner()));
    // A peer may have finished the download while we were blocked on the lock.
    if _guard.is_some()
        && let Some(p) = &cache_path
        && let Ok(bytes) = std::fs::read(p)
        && sha256_hex(&bytes) == d.hash
    {
        drop(_guard);
        downloads.lock().unwrap().remove(&d.hash);
        return Ok(bytes);
    }
    eprintln!("[decryptd] downloading {}", d.filename);

    // Download straight into the cache path when we have a stable key (rsurl
    // manages the sibling `.part` and finalizes atomically); otherwise a temp
    // file we read back and delete. rsurl's URL validator guards the fixed temp
    // path against a stale partial from a different blob.
    let (target, is_temp) = match &cache_path {
        Some(p) => (p.clone(), false),
        None => (cache.join("pending-download.tmp"), true),
    };
    let opts = rsurl::DownloadOptions {
        max_time: Some(Duration::from_secs(300)),
        expected_sha256: parse_sha256_hex(&d.hash),
        ..Default::default()
    };
    rsurl::download(url, &target, opts).map_err(|e| anyhow!("GET {url}: {e}"))?;

    let bytes = std::fs::read(&target)?;
    if is_temp {
        let _ = std::fs::remove_file(&target);
    } else {
        // We just added a finalized blob; keep the cache under its size cap.
        prune_cache(&cache, args.cache_max_gb.saturating_mul(1 << 30));
    }
    // Release the per-hash lock and drop its map entry (a later fetch of the same
    // hash cache-hits, or re-creates the entry) so `downloads` only holds
    // in-flight hashes.
    drop(_guard);
    if !d.hash.is_empty() {
        downloads.lock().unwrap().remove(&d.hash);
    }
    Ok(bytes)
}

/// Keep the blob cache under `max_bytes` by evicting the oldest finalized entries
/// (by mtime). Best-effort — any IO error just leaves that file in place. Skips
/// rsurl's in-progress `.part`/`.tmp` files so an active download is never
/// disturbed; everything else is fair game, since `fetch_blob` reads each blob
/// fully into memory and no running job depends on its file surviving.
fn prune_cache(cache: &Path, max_bytes: u64) {
    let Ok(rd) = std::fs::read_dir(cache) else {
        return;
    };
    let mut entries: Vec<(std::time::SystemTime, u64, PathBuf)> = Vec::new();
    let mut total: u64 = 0;
    for e in rd.flatten() {
        let path = e.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.ends_with(".part") || name.ends_with(".tmp") {
            continue; // an in-progress download rsurl still owns
        }
        let Ok(meta) = e.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
        total = total.saturating_add(meta.len());
        entries.push((mtime, meta.len(), path));
    }
    if total <= max_bytes {
        return;
    }
    entries.sort_by_key(|(mtime, _, _)| *mtime); // oldest first
    for (_, len, path) in entries {
        if total <= max_bytes {
            break;
        }
        if std::fs::remove_file(&path).is_ok() {
            total = total.saturating_sub(len);
            eprintln!("[decryptd] cache: evicted {} ({len} B)", path.display());
        }
    }
}

/// Decode an RFC 2397 `data:` URI into its raw bytes. Handles the two payload
/// encodings: `;base64` (the platform's case — base64 over the gzip/xz blob) and
/// the default percent-encoding. The media type in the header is ignored; the
/// caller's filename/signature detection picks the container format downstream.
fn decode_data_url(url: &str) -> Result<Vec<u8>> {
    use base64::Engine;
    let body = url
        .strip_prefix("data:")
        .ok_or_else(|| anyhow!("not a data: URL"))?;
    // `data:[<mediatype>][;base64],<payload>` — split on the first comma.
    let (header, payload) = body
        .split_once(',')
        .ok_or_else(|| anyhow!("malformed data: URL (no comma)"))?;
    if header
        .rsplit(';')
        .any(|seg| seg.eq_ignore_ascii_case("base64"))
    {
        // Base64 may carry embedded whitespace (line wrapping); strip it first.
        let cleaned: String = payload
            .chars()
            .filter(|c| !c.is_ascii_whitespace())
            .collect();
        // Be liberal in what we accept: the platform sometimes emits the payload
        // without trailing `=` padding (which the strict STANDARD engine rejects
        // as "Invalid padding") and sometimes in the URL-safe `-_` alphabet. Pick
        // the alphabet from the payload and treat padding as optional.
        use base64::alphabet;
        use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
        let alpha = if cleaned.bytes().any(|b| b == b'-' || b == b'_') {
            &alphabet::URL_SAFE
        } else {
            &alphabet::STANDARD
        };
        let cfg =
            GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent);
        GeneralPurpose::new(alpha, cfg)
            .decode(cleaned.as_bytes())
            .map_err(|e| anyhow!("decoding base64 data: URL: {e}"))
    } else {
        percent_decode(payload)
    }
}

/// Percent-decode a (non-base64) `data:` URL payload into raw bytes.
fn percent_decode(s: &str) -> Result<Vec<u8>> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' => {
                let hex = s
                    .get(i + 1..i + 3)
                    .ok_or_else(|| anyhow!("truncated percent-escape in data: URL"))?;
                out.push(
                    u8::from_str_radix(hex, 16)
                        .map_err(|_| anyhow!("invalid percent-escape %{hex} in data: URL"))?,
                );
                i += 3;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    Ok(out)
}

/// Unpack engine.zip: parse `manifest.json` and collect every `*.sm<NN>.cubin` as
/// an `(arch, bytes)` pair, highest compute-capability first. The arch tag rides
/// along so the GPU loader can skip cubins newer than the device (see
/// [`cuda::Gpu::load_first`]) instead of handing them to a driver that may crash.
fn unpack_engine(zip_bytes: &[u8]) -> Result<(Manifest, Vec<Cubin>)> {
    let mut zip = zip::ZipArchive::new(Cursor::new(zip_bytes)).context("opening engine.zip")?;
    let mut manifest: Option<Manifest> = None;
    let mut cubins: Vec<(u32, Vec<u8>)> = Vec::new();
    for i in 0..zip.len() {
        let mut f = zip.by_index(i)?;
        let name = f.name().rsplit('/').next().unwrap_or(f.name()).to_string();
        if name == "manifest.json" {
            let mut s = String::new();
            f.read_to_string(&mut s)?;
            manifest = Some(serde_json::from_str(&s).context("parsing manifest.json")?);
        } else if let Some(i) = name.find(".sm")
            && let Some(rest) = name[i + 3..].strip_suffix(".cubin")
            && let Ok(arch) = rest.parse::<u32>()
        {
            let mut buf = Vec::new();
            f.read_to_end(&mut buf)?;
            cubins.push((arch, buf));
        }
    }
    let manifest = manifest.ok_or_else(|| anyhow!("engine.zip has no manifest.json"))?;
    if cubins.is_empty() {
        bail!("engine.zip contains no *.sm<NN>.cubin files");
    }
    cubins.sort_by_key(|c| std::cmp::Reverse(c.0));
    Ok((manifest, cubins))
}

// --------------------------------------------------------------------------- run
/// A fragment that's been claimed and has its blobs downloaded — ready for the GPU.
struct ReadyJob {
    frag_id: String,
    start: u64,
    end: u64,
    manifest: Manifest,
    /// Arch-tagged cubins, highest arch first (see [`Cubin`]).
    cubins: Vec<Cubin>,
    data: Vec<u8>,
    /// Wall-clock seconds spent fetching this job's blobs (0 on a full cache hit).
    download_secs: f64,
}

/// A fragment that's finished on the GPU — ready to compress + submit.
struct FinishedJob {
    frag_id: String,
    start: u64,
    end: u64,
    record_size: u32,
    output: Vec<u8>,
    /// Seconds the GPU spent on this fragment.
    run_secs: f64,
    /// Seconds spent downloading this job's blobs (carried from [`ReadyJob`]).
    download_secs: f64,
    /// Arch tag of the cubin that ran (`X*10+Y`), reported as `sm<NN>` at submit.
    cubin_arch: u32,
    /// Which GPU ran it: the ordinal and the device name.
    gpu_idx: i32,
    gpu_name: String,
}

/// Fragments currently somewhere in the pipeline (claimed → run → submit), mapping
/// each fragment UUID to its latest `Response_Key`. pullOne rotates the key on every
/// issue, so if the platform re-hands us a fragment we're still running, we refresh the
/// stored key here and submit with it — otherwise the in-flight copy's key is stale and
/// the submit is rejected.
type InFlight = Arc<Mutex<HashMap<String, String>>>;

/// Per-content-hash download locks, shared across every GPU's prefetch thread.
/// Two GPUs claiming fragments of the same job fetch the same blobs; without
/// coordination they'd race writing the same file in the shared cache. The first
/// fetcher of a hash holds its lock and downloads; the rest block, then find the
/// blob already cached. Entries are removed once the download finishes, so the
/// map only ever holds in-flight hashes.
type Downloads = Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>;

/// Shared worker state surfaced to the GUI tray: how many fragments are running
/// on the GPU right now. Cheap atomics, threaded through the pipeline even in
/// the headless build (the tray just never reads it there).
#[derive(Clone, Default)]
struct Status {
    /// Count of fragments currently executing on the GPU (0 = waiting for work).
    active: Arc<AtomicUsize>,
    /// Set by the tray's Pause item; the worker parks at its gates while true.
    paused: Arc<AtomicBool>,
    /// On-disk marker (`<workdir>/paused`) mirroring `paused`, so a pause chosen in
    /// the tray survives the auto-updater's restart instead of coming back running.
    /// `None` until [`Status::bind_pause_file`] wires it up (e.g. the `--once` path
    /// never persists); when `None`, pausing is purely in-memory.
    paused_file: Arc<Mutex<Option<PathBuf>>>,
    /// Why the worker is idle when not running — "no open work", a pull error, or
    /// a fatal stop. Surfaced in the tray so a Windows user (whose console logs are
    /// swallowed by the GUI subsystem) can tell an idle worker from a broken one.
    note: Arc<Mutex<String>>,
    /// Rolling count of work items ("tries") run on the GPU, for a live rate in
    /// the tray. Fed per kernel tile by every GPU, so it aggregates across them.
    tries: Arc<Throughput>,
}

/// Timestamped record of GPU work items, for a ~1-minute average "tries/second".
/// Fed per kernel tile (not per fragment) so the rate stays smooth even while a
/// single multi-hundred-billion-item fragment runs for minutes.
#[derive(Default)]
struct Throughput {
    /// `(instant, cumulative items)` samples spanning the window; the front is the
    /// window anchor, the back the latest total. Bounded to ~[`Self::WINDOW`].
    samples: Mutex<VecDeque<(Instant, u64)>>,
}

impl Throughput {
    const WINDOW: Duration = Duration::from_secs(60);

    /// Add `items` just completed on the GPU (per tile). O(1) amortized.
    fn record(&self, items: u64) {
        if items == 0 {
            return;
        }
        let now = Instant::now();
        let mut s = self.samples.lock().unwrap();
        let cum = s.back().map(|&(_, c)| c).unwrap_or(0) + items;
        s.push_back((now, cum));
        // Keep exactly one sample older than the window as the rate anchor.
        while s.len() > 2
            && s.get(1)
                .is_some_and(|&(t, _)| now.duration_since(t) >= Self::WINDOW)
        {
            s.pop_front();
        }
    }

    /// Average items/second across the retained window (up to ~1 min), or `None`
    /// until two samples span a positive interval. Only read by the GUI tray.
    #[cfg_attr(
        not(all(feature = "gui", any(target_os = "linux", target_os = "windows"))),
        allow(dead_code)
    )]
    fn per_sec(&self) -> Option<f64> {
        let s = self.samples.lock().unwrap();
        let (t0, c0) = *s.front()?;
        let (t1, c1) = *s.back()?;
        let dt = t1.duration_since(t0).as_secs_f64();
        (dt > 0.0).then(|| (c1 - c0) as f64 / dt)
    }
}

impl Status {
    /// True while at least one fragment is running on the GPU. Only read by the
    /// GUI tray; harmless and unused in the headless build.
    #[cfg_attr(
        not(all(feature = "gui", any(target_os = "linux", target_os = "windows"))),
        allow(dead_code)
    )]
    fn is_running(&self) -> bool {
        self.active.load(Ordering::Relaxed) > 0
    }

    /// Whether the tray has paused processing. Only read by the GUI tray.
    #[cfg_attr(
        not(all(feature = "gui", any(target_os = "linux", target_os = "windows"))),
        allow(dead_code)
    )]
    fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    /// Set/clear the pause flag. Only called by the GUI tray. Also persists the
    /// state to the on-disk marker (best-effort) when one is bound, so an
    /// auto-update restart resumes in the same paused/running state the user chose.
    #[cfg_attr(
        not(all(feature = "gui", any(target_os = "linux", target_os = "windows"))),
        allow(dead_code)
    )]
    fn set_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::Relaxed);
        if let Some(path) = self.paused_file.lock().unwrap().as_ref() {
            let res = if paused {
                std::fs::write(path, b"")
            } else {
                match std::fs::remove_file(path) {
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                    other => other,
                }
            };
            if let Err(e) = res {
                eprintln!(
                    "[decryptd] could not persist pause state to {}: {e}",
                    path.display()
                );
            }
        }
    }

    /// Bind the on-disk pause marker at `path` and adopt whatever state it records:
    /// if the file exists we come up paused, matching what the user left running
    /// before an auto-update restart. Called once at startup before the tray reads
    /// the flag. Only the long-running worker binds this (`--once` skips it).
    #[cfg_attr(
        not(all(feature = "gui", any(target_os = "linux", target_os = "windows"))),
        allow(dead_code)
    )]
    fn bind_pause_file(&self, path: PathBuf) {
        self.paused.store(path.exists(), Ordering::Relaxed);
        *self.paused_file.lock().unwrap() = Some(path);
    }

    /// Park the calling worker thread while paused (polled). Compiled into every
    /// build but only ever blocks when the tray flips the flag — the headless
    /// worker never pauses. Cheap: a relaxed load every 200 ms while idle.
    fn wait_while_paused(&self) {
        while self.paused.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(200));
        }
    }

    /// Record `items` work items just run on the GPU (fed per kernel tile).
    fn record_tries(&self, items: u64) {
        self.tries.record(items);
    }

    /// Recent GPU throughput in items/second (~1-minute average), or `None` when
    /// nothing has run recently. Only read by the GUI tray.
    #[cfg_attr(
        not(all(feature = "gui", any(target_os = "linux", target_os = "windows"))),
        allow(dead_code)
    )]
    fn tries_per_sec(&self) -> Option<f64> {
        self.tries.per_sec()
    }

    /// Record why the worker is idle (shown in the tray). Set by the pipeline on
    /// pull errors / no-work and by the GUI on a fatal worker stop.
    fn set_note(&self, note: impl Into<String>) {
        *self.note.lock().unwrap() = note.into();
    }

    /// The current idle note, or empty. Only read by the GUI tray.
    #[cfg_attr(
        not(all(feature = "gui", any(target_os = "linux", target_os = "windows"))),
        allow(dead_code)
    )]
    fn note(&self) -> String {
        self.note.lock().unwrap().clone()
    }

    /// Mark a GPU run as started; the returned guard marks it finished on drop
    /// (so a panic in `run_on_gpu` can't strand the counter above zero).
    fn run_guard(&self) -> RunGuard {
        self.active.fetch_add(1, Ordering::Relaxed);
        RunGuard(self.active.clone())
    }
}

struct RunGuard(Arc<AtomicUsize>);
impl Drop for RunGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Claim the next fragment (`pullOne`) and download its blobs. `Ok(None)` means there
/// was no work — either the platform had none, or it re-issued a fragment we're already
/// processing (in which case the caller should just back off and retry).
fn claim_and_fetch(
    args: &RunArgs,
    downloads: &Downloads,
    worker_id: &str,
    ctx: &RestContext,
    inflight: &InFlight,
) -> Result<Option<ReadyJob>> {
    let resp = ctx
        .do_request(
            "Decrypt/Job:pullOne",
            "POST",
            json!({ "worker": worker_id }),
        )
        .map_err(|e| anyhow!("Decrypt/Job:pullOne: {e}"))?;
    // pullOne returns null data when there are no open jobs with fragments to issue.
    let Some(data) = resp.raw().filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let pull: Pull = serde_json::from_value(data.clone()).context("parsing pullOne response")?;
    let (start, end) = (pull.fragment.range_start, pull.fragment.range_end);
    if end <= start {
        bail!("empty fragment range [{start}, {end})");
    }
    let frag_id = pull.fragment.id.clone();

    // The platform round-robins fragments; with few open fragments it can re-hand us one
    // we're still working on. The pull just rotated its Response_Key (invalidating the
    // copy in flight), so adopt the fresh key, then back off instead of running it twice.
    {
        let mut map = inflight.lock().unwrap();
        if let Some(slot) = map.get_mut(&frag_id) {
            *slot = pull.response_key.clone();
            eprintln!("[decryptd] fragment {frag_id} re-issued; refreshed key, backing off");
            return Ok(None);
        }
        map.insert(frag_id.clone(), pull.response_key.clone());
    }

    // The fragment is now ours; release it from the in-flight set if download fails.
    let fetched = (|| -> Result<ReadyJob> {
        let mut engine_zip: Option<Vec<u8>> = None;
        let mut data_blob: Option<(String, Vec<u8>)> = None;
        // Time just the blob fetches (0 on a full cache hit) — reported at submit.
        let dl_start = Instant::now();
        for d in &pull.job.data {
            let bytes = fetch_blob(args, downloads, d)?;
            if d.filename.ends_with(".zip") {
                engine_zip = Some(bytes);
            } else {
                data_blob = Some((d.filename.clone(), bytes));
            }
        }
        let download_secs = dl_start.elapsed().as_secs_f64();
        let engine_zip = engine_zip.ok_or_else(|| anyhow!("job has no engine .zip blob"))?;
        let (manifest, cubins) = unpack_engine(&engine_zip)?;
        let data = match data_blob {
            Some((name, blob)) => decompress_data(&name, &blob)?,
            None => Vec::new(),
        };
        Ok(ReadyJob {
            frag_id: frag_id.clone(),
            start,
            end,
            manifest,
            cubins,
            data,
            download_secs,
        })
    })();

    match fetched {
        Ok(job) => {
            eprintln!(
                "[decryptd] claimed [{start}, {end}) ({} items)",
                end - start
            );
            Ok(Some(job))
        }
        Err(e) => {
            inflight.lock().unwrap().remove(&frag_id);
            Err(e)
        }
    }
}

/// Decompress a data blob, auto-detecting the container format. The publisher
/// historically shipped `data.xz`, but gzip works too. The filename extension
/// takes priority — `data.gz` is treated as gzip regardless of its bytes — and
/// only when the name is inconclusive (or names a codec we can't build) do we
/// fall back to `factory::detect` sniffing the leading magic bytes. A blob that
/// neither names nor looks like a known format is passed through verbatim,
/// assumed already-plaintext. (Only formats enabled in compcol's features are
/// usable; see `Cargo.toml`.)
fn decompress_data(filename: &str, blob: &[u8]) -> Result<Vec<u8>> {
    // Name first, then content signature. `decoder_by_name` returning `None`
    // (codec not compiled in) lets a recognized-but-unbuildable extension fall
    // through to signature detection rather than hard-erroring.
    let picked = codec_for_extension(filename)
        .and_then(|name| compcol::factory::decoder_by_name(name).map(|dec| (name, dec)))
        .or_else(|| {
            compcol::factory::detect(blob)
                .and_then(|name| compcol::factory::decoder_by_name(name).map(|dec| (name, dec)))
        });
    let Some((name, mut dec)) = picked else {
        return Ok(blob.to_vec());
    };
    decode_stream(dec.as_mut(), blob).map_err(|e| anyhow!("{name}-decompressing data: {e:?}"))
}

/// Map a filename's extension to the compcol codec name that handles it, or
/// `None` for an unknown/absent extension. String literals (not the codecs'
/// `NAME` constants) so this stays valid for formats not compiled in — the
/// caller turns an unbuildable name into a fall-through to signature detection.
fn codec_for_extension(filename: &str) -> Option<&'static str> {
    let ext = filename.rsplit('.').next()?;
    match ext.to_ascii_lowercase().as_str() {
        "xz" => Some("xz"),
        "gz" | "gzip" => Some("gzip"),
        "bz2" | "bzip2" => Some("bzip2"),
        "zst" | "zstd" => Some("zstd"),
        _ => None,
    }
}

/// Drive a runtime-selected [`compcol::Decoder`] to completion over an in-memory
/// input. Mirrors `compcol::vec::decompress_to_vec`, which is generic over a
/// compile-time `Algorithm` and so can't take the boxed decoder `factory`
/// hands back.
fn decode_stream(dec: &mut dyn compcol::Decoder, input: &[u8]) -> Result<Vec<u8>, compcol::Error> {
    use compcol::Status;
    const SCRATCH: usize = 64 * 1024;
    let mut out: Vec<u8> = Vec::with_capacity(input.len().saturating_mul(2));
    let mut scratch = vec![0u8; SCRATCH];

    let mut consumed = 0usize;
    while consumed < input.len() {
        let (p, status) = dec.decode(&input[consumed..], &mut scratch)?;
        out.extend_from_slice(&scratch[..p.written]);
        consumed += p.consumed;
        match status {
            Status::StreamEnd => return Ok(out),
            Status::InputEmpty => break,
            Status::OutputFull => continue,
        }
    }
    loop {
        let (p, status) = dec.finish(&mut scratch)?;
        out.extend_from_slice(&scratch[..p.written]);
        if matches!(status, Status::StreamEnd) {
            break;
        }
        if p.written == 0 {
            return Err(compcol::Error::Corrupt);
        }
    }
    Ok(out)
}

/// Run one ready fragment on GPU `ordinal`. Creates its own CUDA context on the
/// calling thread, so multiple of these run concurrently across `--jobs` runners
/// and across GPUs.
/// Cap on active compute time for a single fragment. A fragment that hasn't
/// finished within this bound is aborted (dropped, not submitted) so a runaway or
/// mis-sized job can't hold a GPU indefinitely; the worker backs off and claims the
/// next. Paused time doesn't count against it (see [`cuda::run_job`]).
const JOB_TIMEOUT: Duration = Duration::from_secs(60 * 60);

fn run_on_gpu(ordinal: i32, job: ReadyJob, status: &Status) -> Result<FinishedJob> {
    let gpu = cuda::Gpu::load_first(ordinal, &job.cubins).map_err(|e| anyhow!(e))?;
    let (maj, min) = gpu.compute_capability();
    let cubin_arch = gpu.cubin_arch();
    let gpu_name = gpu.device_name();
    eprintln!(
        "[decryptd] running [{}, {}) on GPU#{ordinal} {gpu_name} (sm_{maj}{min}, cubin sm{cubin_arch}): entry={}",
        job.start, job.end, job.manifest.entry,
    );
    let t0 = Instant::now();
    let output = cuda::run_job(
        &gpu,
        &job.manifest.entry,
        &job.data,
        job.start,
        job.end - 1, // run_job's range is inclusive
        job.manifest.record_size,
        job.manifest.out_cap,
        job.manifest.block,
        job.manifest.tile,
        JOB_TIMEOUT,
        {
            // Feed the tray's rate meter per tile: record the item delta since the
            // last callback (`done` is cumulative within the fragment).
            let mut reported = 0u64;
            move |done, _total| {
                status.record_tries(done.saturating_sub(reported));
                reported = done;
            }
        },
        // Between tiles, park here while paused so the GPU stops computing
        // promptly and the current fragment resumes losslessly.
        || status.wait_while_paused(),
    )
    .map_err(|e| anyhow!("run_job: {e}"))?;
    let run_secs = t0.elapsed().as_secs_f64();
    let records = output.len() / job.manifest.record_size.max(1) as usize;
    eprintln!(
        "[decryptd] ran [{}, {}): {records} record(s) in {run_secs:.1}s",
        job.start, job.end
    );
    Ok(FinishedJob {
        frag_id: job.frag_id,
        start: job.start,
        end: job.end,
        record_size: job.manifest.record_size,
        output,
        run_secs,
        download_secs: job.download_secs,
        cubin_arch,
        gpu_idx: ordinal,
        gpu_name,
    })
}

/// Round a seconds value to milliseconds and wrap it as a JSON number, so submit
/// telemetry carries `1.234` rather than a long float tail.
fn round_ms(secs: f64) -> Value {
    Value::from((secs * 1000.0).round() / 1000.0)
}

/// Compress a finished fragment's records and submit them via `Decrypt/Job:submit`
/// (the platform's standard upload / prepareCbCtx flow).
fn submit_job(ctx: &RestContext, response_key: &str, job: &FinishedJob) -> Result<()> {
    let records = job.output.len() / job.record_size.max(1) as usize;
    let packed = compcol::vec::compress_to_vec::<compcol::xz::Xz>(&job.output)
        .map_err(|e| anyhow!("xz-compressing result: {e:?}"))?;
    let packed_len = packed.len();
    let mut params: HashMap<String, Value> = HashMap::new();
    params.insert(
        "response_key".to_string(),
        Value::String(response_key.to_string()),
    );
    // Telemetry for the platform: how long the blobs took to fetch, how long the
    // GPU ran, which cubin arch actually executed (sm89, …), and the worker build.
    params.insert("download_seconds".to_string(), round_ms(job.download_secs));
    params.insert("run_seconds".to_string(), round_ms(job.run_secs));
    params.insert(
        "cubin".to_string(),
        Value::String(format!("sm{}", job.cubin_arch)),
    );
    params.insert("gpu_idx".to_string(), Value::from(job.gpu_idx));
    params.insert("gpu_name".to_string(), Value::String(job.gpu_name.clone()));
    params.insert(
        "version".to_string(),
        Value::String(env!("CARGO_PKG_VERSION").to_string()),
    );
    klbfw::upload(
        ctx,
        "Decrypt/Job:submit",
        "POST",
        params,
        Cursor::new(packed),
        "application/x-xz",
        None,
    )
    .map_err(|e| anyhow!("Decrypt/Job:submit: {e}"))?;
    eprintln!(
        "[decryptd] submitted {records} record(s) ({packed_len} B xz) for [{}, {}) (ran {:.1}s, dl {:.1}s)",
        job.start, job.end, job.run_secs, job.download_secs
    );
    Ok(())
}

// ---------------------------------------------------------------- pipeline stages
/// Prefetch stage: keep pulling + downloading the next fragment so a ready job is
/// waiting whenever the GPU frees up.
fn prefetch_loop(
    args: Arc<RunArgs>,
    downloads: Downloads,
    worker_id: String,
    ctx: RestContext,
    inflight: InFlight,
    ready: SyncSender<ReadyJob>,
    status: Status,
) {
    loop {
        // Don't claim new work while paused.
        status.wait_while_paused();
        match claim_and_fetch(&args, &downloads, &worker_id, &ctx, &inflight) {
            Ok(Some(job)) => {
                status.set_note(""); // got work; clear any idle note
                if ready.send(job).is_err() {
                    return; // pipeline shut down
                }
            }
            Ok(None) => {
                eprintln!("[decryptd] no work; sleeping {}s", args.idle_secs);
                status.set_note("no open work");
                thread::sleep(Duration::from_secs(args.idle_secs));
            }
            Err(e) => {
                eprintln!("[decryptd] pull error: {e:#}");
                status.set_note(format!("pull error: {e}"));
                thread::sleep(Duration::from_secs(args.idle_secs));
            }
        }
    }
}

/// GPU stage: the serialized step. One per `--jobs`; each takes a ready job, runs it,
/// and hands the result to the upload stage.
fn run_loop(
    ordinal: i32,
    args: Arc<RunArgs>,
    ready: Arc<Mutex<Receiver<ReadyJob>>>,
    inflight: InFlight,
    done: SyncSender<FinishedJob>,
    status: Status,
) {
    loop {
        let job = match ready.lock().unwrap().recv() {
            Ok(job) => job,
            Err(_) => return, // prefetcher gone
        };
        // Don't start a new fragment (or even load the GPU) while paused.
        status.wait_while_paused();
        let frag_id = job.frag_id.clone();
        // Flip the tray status to "Running" for the duration of this GPU run.
        let _running = status.run_guard();
        match run_on_gpu(ordinal, job, &status) {
            Ok(finished) => {
                if done.send(finished).is_err() {
                    inflight.lock().unwrap().remove(&frag_id);
                    return;
                }
            }
            Err(e) => {
                eprintln!("[decryptd] run error: {e:#}");
                inflight.lock().unwrap().remove(&frag_id);
                // Back off before taking the next fragment. Without this a
                // persistent GPU fault (OOM, driver wedged, no compatible cubin)
                // spins here, claiming + downloading fragments as fast as the
                // network allows and burning the work pool for nothing.
                thread::sleep(Duration::from_secs(args.idle_secs));
            }
        }
    }
}

/// Upload stage: compress + submit finished results in the background while the GPU
/// runs the next job.
fn upload_loop(ctx: RestContext, inflight: InFlight, done: Arc<Mutex<Receiver<FinishedJob>>>) {
    loop {
        let job = match done.lock().unwrap().recv() {
            Ok(job) => job,
            Err(_) => return,
        };
        // Read the latest key (the prefetcher may have refreshed it on a re-issue).
        let key = inflight.lock().unwrap().get(&job.frag_id).cloned();
        match key {
            Some(key) => {
                if let Err(e) = submit_job(&ctx, &key, &job) {
                    eprintln!(
                        "[decryptd] submit error for [{}, {}): {e:#}",
                        job.start, job.end
                    );
                }
            }
            None => eprintln!(
                "[decryptd] no response key for fragment {}; dropping result",
                job.frag_id
            ),
        }
        inflight.lock().unwrap().remove(&job.frag_id);
    }
}

/// Trust anchor for self-updates: the SHA-256 fingerprint of the decryptd
/// release signing key (`rsupd id export`). It's a hash of a public key, so it's
/// safe to embed; the updater refuses any manifest not signed by the matching
/// private identity.
const RSUPD_FINGERPRINT: &str = "80b9edc7e6eaebf10b2a25bb10556b9b7fa6abc9fbe556706a2b680cefa4a0fc";

/// Build the signed auto-updater. The transport (dist-go over rsurl) and channel
/// (`master`) default from the fingerprint, so the anchor is the only required
/// input. The git stamps from `build.rs` let it also spot a newer build of the
/// same version (and never reinstall the identical build). Every release is one
/// build (the tray degrades to headless via ldtray), so there's a single update
/// stream for all workers.
fn build_updater() -> rsupd::Result<rsupd::Updater> {
    rsupd::Updater::builder(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
        .fingerprint_hex(RSUPD_FINGERPRINT)
        .git_tag(env!("RSUPD_GIT_TAG"))
        .date_tag(rsupd::date_tag_from_unix(env!("RSUPD_BUILD_UNIX")))
        .build()
}

fn main() -> Result<()> {
    // If we were just re-exec'd by a self-update, settle briefly before running.
    rsupd::honor_startup_delay();

    let args = RunArgs::parse();
    let status = Status::default();

    // Long-lived workers keep themselves current: check hourly in the background
    // and restart into each new signed build. `--once` is short-lived, so it skips
    // the updater. This is also where the pause state is made durable: the tray can
    // pause the worker, and without this an auto-update restart would resume running.
    if !args.once {
        // Bind the on-disk pause marker before anything reads the flag, adopting the
        // paused/running state a previous run (pre-restart) left behind. The workdir
        // is created here too so the marker has somewhere to live (run_worker also
        // creates it, idempotently). Bound wherever a pause control exists: any Unix
        // build (the SIGUSR1 toggle below) and the Windows tray build. Windows-
        // headless has no way to pause, so it neither adopts nor writes a marker.
        #[cfg(any(unix, all(feature = "gui", target_os = "windows")))]
        {
            let workdir = args.workdir();
            if let Err(e) = std::fs::create_dir_all(&workdir) {
                eprintln!(
                    "[decryptd] cannot prepare workdir {} for pause state: {e}",
                    workdir.display()
                );
            }
            status.bind_pause_file(workdir.join("paused"));
        }

        // On Unix, SIGUSR1 toggles pause/resume: the only pause control on a headless
        // server, and a scriptable one on the desktop (`kill -USR1 <pid>`). The toggle
        // goes through `set_paused`, so it persists to the marker just like the tray.
        #[cfg(unix)]
        {
            let status = status.clone();
            match signal_hook::iterator::Signals::new([signal_hook::consts::SIGUSR1]) {
                Ok(mut signals) => {
                    thread::spawn(move || {
                        for _ in signals.forever() {
                            let paused = !status.is_paused();
                            status.set_paused(paused);
                            eprintln!(
                                "[decryptd] SIGUSR1: {}",
                                if paused { "paused" } else { "resumed" }
                            );
                        }
                    });
                }
                Err(e) => eprintln!("[decryptd] SIGUSR1 pause toggle unavailable: {e}"),
            }
        }

        match build_updater() {
            Ok(updater) => {
                updater.spawn_auto_update(false);
            }
            Err(e) => eprintln!("[decryptd] auto-update disabled: {e}"),
        }
    }

    // GUI build (Windows/Linux): unless `--once`, hand the main thread to the
    // system-tray event loop and run the worker pipeline behind it.
    #[cfg(all(feature = "gui", any(target_os = "linux", target_os = "windows")))]
    if !args.once {
        return gui::run_with_tray(args, status);
    }

    run_worker(args, status)
}

/// Resolve which GPU ordinals to run on: the `--gpus` list if given (validated
/// against the detected count), otherwise every detected GPU.
fn select_gpus(spec: &Option<String>, count: i32) -> Result<Vec<i32>> {
    let Some(spec) = spec else {
        return Ok((0..count).collect());
    };
    let mut ords = Vec::new();
    for tok in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let ord: i32 = tok
            .parse()
            .map_err(|_| anyhow!("--gpus: '{tok}' is not a GPU ordinal"))?;
        if ord < 0 || ord >= count {
            bail!("--gpus: ordinal {ord} out of range (have {count} GPU(s): 0..{count})");
        }
        if !ords.contains(&ord) {
            ords.push(ord);
        }
    }
    if ords.is_empty() {
        bail!("--gpus selected no GPUs");
    }
    Ok(ords)
}

/// Load this worker's persistent id from `<workdir>/worker-id`, or mint a new
/// UUIDv7 and save it. Sent as `worker=<id>` on every pullOne so the platform can
/// attribute a worker's fragments across restarts. Any non-empty existing content
/// is kept verbatim (so an operator can pin a chosen id); only a missing or blank
/// file triggers generation.
fn load_or_create_worker_id(workdir: &Path) -> Result<String> {
    let path = workdir.join("worker-id");
    if let Ok(existing) = std::fs::read_to_string(&path) {
        let existing = existing.trim();
        if !existing.is_empty() {
            return Ok(existing.to_string());
        }
    }
    let id = Uuid::now_v7().to_string();
    std::fs::write(&path, &id).with_context(|| format!("writing {}", path.display()))?;
    eprintln!("[decryptd] generated worker id {id} ({})", path.display());
    Ok(id)
}

fn run_worker(args: RunArgs, status: Status) -> Result<()> {
    let workdir = args.workdir();
    std::fs::create_dir_all(&workdir)
        .with_context(|| format!("creating workdir {}", workdir.display()))?;
    let ctx = RestContext::with_config(Config::new("https".to_string(), args.host.clone()))
        .with_debug(std::env::var("DECRYPTD_DEBUG").is_ok());
    let jobs = args.jobs.max(1);
    let worker_id = load_or_create_worker_id(&workdir)?;

    let count = cuda::device_count().map_err(|e| anyhow!("enumerating GPUs: {e}"))?;
    if count < 1 {
        bail!("no CUDA devices found");
    }
    let gpus = select_gpus(&args.gpus, count)?;

    let inflight: InFlight = Arc::new(Mutex::new(HashMap::new()));
    let downloads: Downloads = Arc::new(Mutex::new(HashMap::new()));

    if args.once {
        // Single fragment on the first selected GPU.
        let ord = gpus[0];
        return match claim_and_fetch(&args, &downloads, &worker_id, &ctx, &inflight)? {
            Some(job) => {
                let key = inflight.lock().unwrap().get(&job.frag_id).cloned();
                let key = key.ok_or_else(|| anyhow!("fragment lost its response key"))?;
                submit_job(&ctx, &key, &run_on_gpu(ord, job, &status)?)
            }
            None => {
                eprintln!("[decryptd] no work; exiting (--once)");
                Ok(())
            }
        };
    }

    eprintln!(
        "[decryptd] host={} worker={worker_id} GPUs={gpus:?} jobs={jobs}/GPU ({} runner(s) total)",
        args.host,
        gpus.len() * jobs
    );

    let args = Arc::new(args);
    let mut runners = Vec::new();

    // One independent pipeline per GPU: its own prefetch → bounded ready queue →
    // `jobs` runners pinned to that GPU → `jobs` uploaders. Sharing `inflight`
    // (dedupes a fragment the platform hands to two GPUs) and `downloads`
    // (dedupes a blob two GPUs fetch into the shared cache).
    for &ord in &gpus {
        let (ready_tx, ready_rx) = sync_channel::<ReadyJob>(jobs);
        let (done_tx, done_rx) = sync_channel::<FinishedJob>(jobs);
        let ready_rx = Arc::new(Mutex::new(ready_rx));
        let done_rx = Arc::new(Mutex::new(done_rx));

        {
            let (args, downloads, ctx, inflight, status, worker_id) = (
                args.clone(),
                downloads.clone(),
                ctx.clone(),
                inflight.clone(),
                status.clone(),
                worker_id.clone(),
            );
            thread::spawn(move || {
                prefetch_loop(args, downloads, worker_id, ctx, inflight, ready_tx, status)
            });
        }
        for _ in 0..jobs {
            let (ctx, inflight, done_rx) = (ctx.clone(), inflight.clone(), done_rx.clone());
            thread::spawn(move || upload_loop(ctx, inflight, done_rx));
        }
        for _ in 0..jobs {
            let (args, ready_rx, inflight, done_tx) = (
                args.clone(),
                ready_rx.clone(),
                inflight.clone(),
                done_tx.clone(),
            );
            let status = status.clone();
            runners.push(thread::spawn(move || {
                run_loop(ord, args, ready_rx, inflight, done_tx, status)
            }));
        }
        drop(done_tx);
    }

    // Runner threads loop forever; joining them blocks until the process is killed.
    for r in runners {
        let _ = r.join();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // The exact inline blob the platform handed back in the field, which rsurl
    // rejected as an "invalid URL": a gzip stream of a `PCB1`-tagged data blob.
    const SAMPLE: &str = "data:application/gzip;base64,H4sIAAAAAAAA/wtwdjJkAAJ2II4DYg0GCLjZF1uzaFNdxUHdh20rfjCvjDa0uNC6SVJlcaczH+MpkezyD6pK2ySFYhWVlFVU1dQ1NLW0dXT19A0MjYxNTM3MLSytrG1s7ewdHJ2cXVzd3D08vbx9fP38AwKDgkNCw8IjIqOiY2Lj4hMSk5JTUtPSMzKzsnNy8/ILCouKS0rLyisqq6prausAYVRJS54AAAA=";

    #[test]
    fn data_url_base64_gzip_roundtrips() {
        let gz = decode_data_url(SAMPLE).expect("decode");
        // Decodes to a real gzip stream (magic 1f 8b) that our data-blob
        // decompressor unpacks to the `PCB1`-tagged payload.
        assert_eq!(&gz[..2], &[0x1f, 0x8b], "gzip magic");
        let plain = decompress_data("blob.gz", &gz).expect("gunzip");
        assert_eq!(&plain[..4], b"PCB1", "unpacked blob tag");
    }

    #[test]
    fn data_url_percent_encoded() {
        let bytes = decode_data_url("data:text/plain,a%20b%2Fc").expect("decode");
        assert_eq!(bytes, b"a b/c");
    }

    #[test]
    fn data_url_base64_case_insensitive_and_whitespace() {
        // ";BASE64" and wrapped payload both tolerated.
        let bytes =
            decode_data_url("data:application/octet-stream;BASE64,aGVs\nbG8=").expect("decode");
        assert_eq!(bytes, b"hello");
    }

    #[test]
    fn data_url_base64_unpadded() {
        // "hello" is 5 bytes → base64 "aGVsbG8=" with padding. The platform
        // sometimes drops the trailing `=`; we must still decode it.
        let bytes =
            decode_data_url("data:application/octet-stream;base64,aGVsbG8").expect("decode");
        assert_eq!(bytes, b"hello");
    }

    #[test]
    fn data_url_base64_url_safe_alphabet() {
        // 0xff 0xfe → standard "//4=", url-safe "__4" (unpadded). Exercises
        // both the `-_` alphabet and missing padding at once.
        let bytes = decode_data_url("data:application/octet-stream;base64,__4").expect("decode");
        assert_eq!(bytes, &[0xff, 0xfe]);
    }

    #[test]
    fn workdir_override_beats_default() {
        // Explicit --workdir is always honored verbatim.
        let a = RunArgs::parse_from(["decryptd", "--workdir", "/tmp/custom-wd"]);
        assert_eq!(a.workdir(), PathBuf::from("/tmp/custom-wd"));
        // With no flag, workdir() resolves to the default.
        let b = RunArgs::parse_from(["decryptd"]);
        assert_eq!(b.workdir(), default_workdir());
        // Without the tray the default is the CWD-relative folder; the gui build
        // uses a per-user data dir (env-dependent, so not asserted here).
        #[cfg(not(feature = "gui"))]
        assert_eq!(default_workdir(), PathBuf::from("decryptd-data"));
    }

    #[test]
    fn worker_id_persists_and_is_reused() {
        let dir = std::env::temp_dir().join(format!("decryptd-wid-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // First call mints a UUIDv7 and writes it.
        let id1 = load_or_create_worker_id(&dir).unwrap();
        assert_eq!(id1.len(), 36, "UUID string form");
        assert_eq!(Uuid::parse_str(&id1).unwrap().get_version_num(), 7);
        // Second call returns the same persisted id (no regeneration).
        let id2 = load_or_create_worker_id(&dir).unwrap();
        assert_eq!(id1, id2);

        // A pinned (non-UUID) id is honored verbatim, trimmed of whitespace.
        std::fs::write(dir.join("worker-id"), "  my-pinned-id\n").unwrap();
        assert_eq!(load_or_create_worker_id(&dir).unwrap(), "my-pinned-id");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pause_state_survives_across_binds() {
        let dir = std::env::temp_dir().join(format!("decryptd-pause-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join("paused");

        // A fresh worker with no marker comes up running, and toggling the flag off
        // and on again leaves no stray file / writes the marker on demand.
        let a = Status::default();
        a.bind_pause_file(marker.clone());
        assert!(!a.is_paused(), "no marker → running");
        assert!(!marker.exists());

        // Pausing writes the on-disk marker; resuming removes it.
        a.set_paused(true);
        assert!(marker.exists(), "pause must persist a marker");
        a.set_paused(false);
        assert!(!marker.exists(), "resume must clear the marker");
        // Clearing an already-cleared state is a no-op, not an error.
        a.set_paused(false);
        assert!(!marker.exists());

        // Pause, then simulate an auto-update restart: a brand-new Status binding the
        // same file must adopt the paused state instead of resuming.
        a.set_paused(true);
        let restarted = Status::default();
        assert!(!restarted.is_paused(), "default is running until bound");
        restarted.bind_pause_file(marker.clone());
        assert!(
            restarted.is_paused(),
            "a persisted pause must survive a restart"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // Live NVML probe on real hardware (needs a GPU + driver). Run manually:
    //   cargo test -- --ignored --nocapture nvml_reads
    #[cfg(all(feature = "gui", any(target_os = "linux", target_os = "windows")))]
    #[test]
    #[ignore]
    fn nvml_reads_gpu_telemetry() {
        let pci = crate::cuda::pci_bus_id(0).expect("pci bus id for GPU 0");
        println!("GPU0 pci = {pci}");
        let nvml = crate::nvml::Nvml::load().expect("load NVML");
        let t = nvml.telemetry(&pci);
        println!("temp = {:?} °C, power = {:?} W", t.temp_c, t.power_w);
        assert!(
            t.temp_c.is_some() || t.power_w.is_some(),
            "expected NVML to report at least one metric"
        );
    }

    #[test]
    fn throughput_reports_a_positive_rate() {
        let t = Throughput::default();
        assert!(t.per_sec().is_none(), "no rate before any samples");
        t.record(0); // ignored
        t.record(1_000_000);
        thread::sleep(Duration::from_millis(30));
        t.record(1_000_000);
        let r = t.per_sec().expect("a rate after two spaced samples");
        // ~2M items over ~30ms; timing is loose, so only assert it's a sane rate.
        assert!(r > 0.0, "rate should be positive, got {r}");
        // The window keeps the deque bounded — a burst of samples collapses to the
        // anchor + recents, never unbounded.
        for _ in 0..1000 {
            t.record(1);
        }
        assert!(t.samples.lock().unwrap().len() <= 1002);
    }

    #[test]
    fn pause_gate_blocks_until_resumed() {
        let status = Status::default();
        assert!(!status.is_paused());
        // Not paused: the gate returns immediately.
        status.wait_while_paused();

        status.set_paused(true);
        assert!(status.is_paused());
        let resumer = status.clone();
        let t = thread::spawn(move || {
            thread::sleep(Duration::from_millis(150));
            resumer.set_paused(false);
        });
        let t0 = Instant::now();
        status.wait_while_paused(); // blocks until the resumer clears the flag
        let waited = t0.elapsed();
        t.join().unwrap();
        assert!(!status.is_paused());
        assert!(
            waited >= Duration::from_millis(100),
            "gate returned before resume: {waited:?}"
        );
    }

    #[test]
    fn select_gpus_defaults_and_validates() {
        // Default: every detected GPU.
        assert_eq!(select_gpus(&None, 3).unwrap(), vec![0, 1, 2]);
        // Explicit list, order preserved, duplicates collapsed.
        assert_eq!(select_gpus(&Some("2,0,2".into()), 4).unwrap(), vec![2, 0]);
        // Whitespace tolerated.
        assert_eq!(select_gpus(&Some(" 1 , 0 ".into()), 2).unwrap(), vec![1, 0]);
        // Out of range and non-numeric are rejected.
        assert!(select_gpus(&Some("3".into()), 2).is_err());
        assert!(select_gpus(&Some("x".into()), 2).is_err());
        assert!(select_gpus(&Some("".into()), 2).is_err());
    }

    #[test]
    fn prune_cache_evicts_down_to_cap_and_spares_in_progress() {
        // Unique scratch dir so parallel test runs don't collide.
        let dir = std::env::temp_dir().join(format!("decryptd-prune-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Three finalized 10-byte blobs (30 B) plus an in-progress .part that
        // eviction must never touch even though we're over the cap.
        for name in ["aaa", "bbb", "ccc"] {
            std::fs::write(dir.join(name), [0u8; 10]).unwrap();
        }
        std::fs::write(dir.join("pending-download.tmp.part"), [0u8; 10]).unwrap();

        // Cap at 15 B: must evict finalized blobs until <= 15 (keeps exactly one),
        // leaving the .part alone.
        prune_cache(&dir, 15);

        let finalized = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| !n.ends_with(".part"))
            .count();
        assert_eq!(finalized, 1, "should evict down to one finalized blob");
        assert!(
            dir.join("pending-download.tmp.part").exists(),
            "in-progress .part must survive eviction"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
