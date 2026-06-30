//! `decryptd` — a generic volunteer GPU job runner for decrypt.
//!
//! decryptd knows nothing about bloom filters, RNG, or BIP39. It just:
//!   1. claims a fragment of work from the platform (`Decrypt/Job:pullOne`),
//!   2. downloads the job's blobs (`engine.zip` + an optional compressed `data`
//!      blob — xz/gzip/bzip2/zstd, auto-detected) from the inline URLs the pull
//!      response carries,
//!   3. reads launch parameters from `manifest.json` inside `engine.zip`, loads the
//!      cubin for the local GPU, and launches its kernel over the fragment range
//!      (the kernel does all the real work and writes output records),
//!   4. gathers the output records, compresses them (xz), and submits them back with
//!      `Decrypt/Job:submit`.
//!
//! These stages are pipelined: the next fragment downloads while the GPU runs the
//! current one, and finished results upload while the GPU runs the next. Only the GPU
//! run itself is serialized (one at a time unless `--jobs > 1`).
//!
//! `pullOne`/`submit` and the `Decrypt/Data` blobs are anonymous-accessible, so a
//! worker needs no credentials. All job-specific logic lives in the cubin + the data
//! blob, produced by the job's publisher. See the kernel ABI in `cuda.rs`.

mod cuda;

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use klbfw::{Config, RestContext};
use serde::Deserialize;
use serde_json::{Value, json};

/// Run as a worker: claim Decrypt/Job fragments, run the kernel, submit results.
#[derive(Parser)]
#[command(name = "decryptd", about = "Generic GPU job runner for decrypt")]
struct RunArgs {
    /// KarpelesLab platform host (pullOne/submitFragment are anonymous — no key needed).
    #[arg(long, env = "DECRYPTD_HOST", default_value = "www.atonline.com")]
    host: String,
    /// Working directory for the blob cache and scratch.
    #[arg(long, env = "DECRYPTD_WORKDIR", default_value = "decryptd-data")]
    workdir: PathBuf,
    /// Claim a single fragment then exit (default: loop forever).
    #[arg(long)]
    once: bool,
    /// Seconds to wait before retrying when no work is available.
    #[arg(long, default_value_t = 60)]
    idle_secs: u64,
    /// Number of GPU jobs to run concurrently (default: one at a time). Download of
    /// the next job and upload of finished results always overlap the GPU run.
    #[arg(long, default_value_t = 1)]
    jobs: usize,
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

/// Download a Job.Data blob from its inline URL, caching it under its content hash.
fn fetch_blob(args: &RunArgs, d: &DataRef) -> Result<Vec<u8>> {
    let url = d
        .url
        .as_deref()
        .ok_or_else(|| anyhow!("Job.Data entry has no Url"))?;
    let cache = args.workdir.join("cache");
    std::fs::create_dir_all(&cache)?;
    // `Hash` is the blob's content SHA-256, so it doubles as a cache key + checksum.
    let cache_path = (!d.hash.is_empty()).then(|| cache.join(&d.hash));
    if let Some(p) = &cache_path
        && let Ok(bytes) = std::fs::read(p)
        && sha256_hex(&bytes) == d.hash
    {
        return Ok(bytes); // cache hit
    }
    eprintln!("[decryptd] downloading {}", d.filename);
    let resp = rsurl::Request::new("GET", url)
        .map_err(|e| anyhow!("{e}"))?
        .max_time(Duration::from_secs(300))
        .send()
        .map_err(|e| anyhow!("GET {url}: {e}"))?;
    if resp.status >= 400 {
        bail!("GET {url} → HTTP {}", resp.status);
    }
    if !d.hash.is_empty() {
        let got = sha256_hex(&resp.body);
        if got != d.hash {
            bail!("sha256 mismatch for {url}: got {got}, want {}", d.hash);
        }
    }
    if let Some(p) = &cache_path {
        std::fs::write(p, &resp.body)?;
    }
    Ok(resp.body)
}

/// Unpack engine.zip: parse `manifest.json` and collect every `*.sm<NN>.cubin`'s
/// bytes, highest compute-capability first.
fn unpack_engine(zip_bytes: &[u8]) -> Result<(Manifest, Vec<Vec<u8>>)> {
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
    Ok((manifest, cubins.into_iter().map(|(_, b)| b).collect()))
}

// --------------------------------------------------------------------------- run
/// A fragment that's been claimed and has its blobs downloaded — ready for the GPU.
struct ReadyJob {
    frag_id: String,
    start: u64,
    end: u64,
    manifest: Manifest,
    cubins: Vec<Vec<u8>>,
    data: Vec<u8>,
}

/// A fragment that's finished on the GPU — ready to compress + submit.
struct FinishedJob {
    frag_id: String,
    start: u64,
    end: u64,
    record_size: u32,
    output: Vec<u8>,
    run_secs: f64,
}

/// Fragments currently somewhere in the pipeline (claimed → run → submit), mapping
/// each fragment UUID to its latest `Response_Key`. pullOne rotates the key on every
/// issue, so if the platform re-hands us a fragment we're still running, we refresh the
/// stored key here and submit with it — otherwise the in-flight copy's key is stale and
/// the submit is rejected.
type InFlight = Arc<Mutex<HashMap<String, String>>>;

/// Claim the next fragment (`pullOne`) and download its blobs. `Ok(None)` means there
/// was no work — either the platform had none, or it re-issued a fragment we're already
/// processing (in which case the caller should just back off and retry).
fn claim_and_fetch(
    args: &RunArgs,
    ctx: &RestContext,
    inflight: &InFlight,
) -> Result<Option<ReadyJob>> {
    let resp = ctx
        .do_request("Decrypt/Job:pullOne", "POST", json!({}))
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
        let mut data_blob: Option<Vec<u8>> = None;
        for d in &pull.job.data {
            let bytes = fetch_blob(args, d)?;
            if d.filename.ends_with(".zip") {
                engine_zip = Some(bytes);
            } else {
                data_blob = Some(bytes);
            }
        }
        let engine_zip = engine_zip.ok_or_else(|| anyhow!("job has no engine .zip blob"))?;
        let (manifest, cubins) = unpack_engine(&engine_zip)?;
        let data = match data_blob {
            Some(blob) => decompress_data(&blob)?,
            None => Vec::new(),
        };
        Ok(ReadyJob {
            frag_id: frag_id.clone(),
            start,
            end,
            manifest,
            cubins,
            data,
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

/// Decompress a data blob, auto-detecting the container format from its magic
/// bytes. The publisher historically shipped `data.xz`, but any of the common
/// stream formats below works just as well, so we sniff rather than assume.
/// An uncompressed blob (no recognized magic) is passed through verbatim.
fn decompress_data(blob: &[u8]) -> Result<Vec<u8>> {
    let starts = |magic: &[u8]| blob.starts_with(magic);
    if starts(&[0xFD, b'7', b'z', b'X', b'Z', 0x00]) {
        compcol::vec::decompress_to_vec::<compcol::xz::Xz>(blob)
            .map_err(|e| anyhow!("xz-decompressing data: {e:?}"))
    } else if starts(&[0x1F, 0x8B]) {
        compcol::vec::decompress_to_vec::<compcol::gzip::Gzip>(blob)
            .map_err(|e| anyhow!("gzip-decompressing data: {e:?}"))
    } else if starts(b"BZh") {
        compcol::vec::decompress_to_vec::<compcol::bzip2::Bzip2>(blob)
            .map_err(|e| anyhow!("bzip2-decompressing data: {e:?}"))
    } else if starts(&[0x28, 0xB5, 0x2F, 0xFD]) {
        compcol::vec::decompress_to_vec::<compcol::zstd::Zstd>(blob)
            .map_err(|e| anyhow!("zstd-decompressing data: {e:?}"))
    } else {
        // No recognized magic — assume the blob is already plaintext.
        Ok(blob.to_vec())
    }
}

/// Run one ready fragment on the GPU. Creates its own CUDA context, so multiple of
/// these can run concurrently (`--jobs > 1`).
fn run_on_gpu(job: ReadyJob) -> Result<FinishedJob> {
    let gpu = cuda::Gpu::load_first(&job.cubins).map_err(|e| anyhow!(e))?;
    let (maj, min) = gpu.compute_capability();
    eprintln!(
        "[decryptd] running [{}, {}) on {} (sm_{maj}{min}): entry={}",
        job.start,
        job.end,
        gpu.device_name(),
        job.manifest.entry,
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
        |_done, _total| {},
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
    })
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
        "[decryptd] submitted {records} record(s) ({packed_len} B xz) for [{}, {}) (ran {:.1}s)",
        job.start, job.end, job.run_secs
    );
    Ok(())
}

// ---------------------------------------------------------------- pipeline stages
/// Prefetch stage: keep pulling + downloading the next fragment so a ready job is
/// waiting whenever the GPU frees up.
fn prefetch_loop(
    args: Arc<RunArgs>,
    ctx: RestContext,
    inflight: InFlight,
    ready: SyncSender<ReadyJob>,
) {
    loop {
        match claim_and_fetch(&args, &ctx, &inflight) {
            Ok(Some(job)) => {
                if ready.send(job).is_err() {
                    return; // pipeline shut down
                }
            }
            Ok(None) => {
                eprintln!("[decryptd] no work; sleeping {}s", args.idle_secs);
                thread::sleep(Duration::from_secs(args.idle_secs));
            }
            Err(e) => {
                eprintln!("[decryptd] pull error: {e:#}");
                thread::sleep(Duration::from_secs(args.idle_secs));
            }
        }
    }
}

/// GPU stage: the serialized step. One per `--jobs`; each takes a ready job, runs it,
/// and hands the result to the upload stage.
fn run_loop(
    ready: Arc<Mutex<Receiver<ReadyJob>>>,
    inflight: InFlight,
    done: SyncSender<FinishedJob>,
) {
    loop {
        let job = match ready.lock().unwrap().recv() {
            Ok(job) => job,
            Err(_) => return, // prefetcher gone
        };
        let frag_id = job.frag_id.clone();
        match run_on_gpu(job) {
            Ok(finished) => {
                if done.send(finished).is_err() {
                    inflight.lock().unwrap().remove(&frag_id);
                    return;
                }
            }
            Err(e) => {
                eprintln!("[decryptd] run error: {e:#}");
                inflight.lock().unwrap().remove(&frag_id);
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

fn main() -> Result<()> {
    run_worker(RunArgs::parse())
}

fn run_worker(args: RunArgs) -> Result<()> {
    std::fs::create_dir_all(&args.workdir)?;
    let ctx = RestContext::with_config(Config::new("https".to_string(), args.host.clone()))
        .with_debug(std::env::var("DECRYPTD_DEBUG").is_ok());
    let jobs = args.jobs.max(1);
    eprintln!("[decryptd] host={} jobs={jobs}", args.host);

    let inflight: InFlight = Arc::new(Mutex::new(HashMap::new()));

    if args.once {
        return match claim_and_fetch(&args, &ctx, &inflight)? {
            Some(job) => {
                let key = inflight.lock().unwrap().get(&job.frag_id).cloned();
                let key = key.ok_or_else(|| anyhow!("fragment lost its response key"))?;
                submit_job(&ctx, &key, &run_on_gpu(job)?)
            }
            None => {
                eprintln!("[decryptd] no work; exiting (--once)");
                Ok(())
            }
        };
    }

    // Pipeline: prefetch → GPU run (×jobs) → upload (×jobs). The bounded channels keep
    // at most `jobs` fragments waiting at each stage, so we don't over-claim work.
    let (ready_tx, ready_rx) = sync_channel::<ReadyJob>(jobs);
    let (done_tx, done_rx) = sync_channel::<FinishedJob>(jobs);
    let ready_rx = Arc::new(Mutex::new(ready_rx));
    let done_rx = Arc::new(Mutex::new(done_rx));
    let args = Arc::new(args);

    {
        let (args, ctx, inflight) = (args.clone(), ctx.clone(), inflight.clone());
        thread::spawn(move || prefetch_loop(args, ctx, inflight, ready_tx));
    }
    for _ in 0..jobs {
        let (ctx, inflight, done_rx) = (ctx.clone(), inflight.clone(), done_rx.clone());
        thread::spawn(move || upload_loop(ctx, inflight, done_rx));
    }
    let mut runners = Vec::new();
    for _ in 0..jobs {
        let (ready_rx, inflight, done_tx) = (ready_rx.clone(), inflight.clone(), done_tx.clone());
        runners.push(thread::spawn(move || run_loop(ready_rx, inflight, done_tx)));
    }
    drop(done_tx);

    // Runner threads loop forever; joining them blocks until the process is killed.
    for r in runners {
        let _ = r.join();
    }
    Ok(())
}
