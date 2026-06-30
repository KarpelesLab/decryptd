//! `decryptd` — a generic volunteer GPU job runner for decrypt.
//!
//! decryptd knows nothing about bloom filters, RNG, or BIP39. It just:
//!   1. claims a fragment of work from the platform (`Decrypt/Job:pullOne`),
//!   2. downloads the job's blobs (`engine.zip` + an optional `data.xz`) from the
//!      inline URLs the pull response carries,
//!   3. reads launch parameters from `manifest.json` inside `engine.zip`, loads the
//!      cubin for the local GPU, and launches its kernel over the fragment range
//!      (the kernel does all the real work and writes output records),
//!   4. gathers the output records, compresses them (xz), and submits them back with
//!      `Decrypt/Job:submitFragment`.
//!
//! `pullOne`/`submitFragment` and the `Decrypt/Data` blobs are anonymous-accessible,
//! so a worker needs no credentials. All job-specific logic lives in the cubin + the
//! data blob, produced by the job's publisher. See the kernel ABI in `cuda.rs`.

mod cuda;

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::PathBuf;
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
    /// Original filename — `engine.zip` vs `data.*.xz` tells the two blobs apart.
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
/// Claim and process one fragment. `Ok(false)` means the platform had no open work.
fn run_once(args: &RunArgs, ctx: &RestContext) -> Result<bool> {
    let resp = ctx
        .do_request("Decrypt/Job:pullOne", "POST", json!({}))
        .map_err(|e| anyhow!("Decrypt/Job:pullOne: {e}"))?;
    // pullOne returns null data when there are no open jobs with fragments to issue.
    let Some(data) = resp.raw().filter(|v| !v.is_null()) else {
        return Ok(false);
    };
    let pull: Pull = serde_json::from_value(data.clone()).context("parsing pullOne response")?;

    // Download the job's blobs; the filename alone tells engine.zip from the data blob.
    let mut engine_zip: Option<Vec<u8>> = None;
    let mut data_xz: Option<Vec<u8>> = None;
    for d in &pull.job.data {
        let bytes = fetch_blob(args, d)?;
        if d.filename.ends_with(".zip") {
            engine_zip = Some(bytes);
        } else {
            data_xz = Some(bytes);
        }
    }
    let engine_zip = engine_zip.ok_or_else(|| anyhow!("job has no engine .zip blob"))?;
    let (manifest, cubins) = unpack_engine(&engine_zip)?;
    let data = match data_xz {
        Some(xz) => compcol::vec::decompress_to_vec::<compcol::xz::Xz>(&xz)
            .map_err(|e| anyhow!("xz-decompressing data: {e:?}"))?,
        None => Vec::new(),
    };

    let (start, end) = (pull.fragment.range_start, pull.fragment.range_end);
    if end <= start {
        bail!("empty fragment range [{start}, {end})");
    }
    let total = end - start;

    let gpu = cuda::Gpu::load_first(&cubins).map_err(|e| anyhow!(e))?;
    let (maj, min) = gpu.compute_capability();
    eprintln!(
        "[decryptd] fragment [{start}, {end}) on {} (sm_{maj}{min}): entry={} ({total} items)",
        gpu.device_name(),
        manifest.entry,
    );

    let t0 = Instant::now();
    let out = cuda::run_job(
        &gpu,
        &manifest.entry,
        &data,
        start,
        end - 1, // run_job's range is inclusive
        manifest.record_size,
        manifest.out_cap,
        manifest.block,
        manifest.tile,
        |done, total| {
            eprint!(
                "\r  {done}/{total} ({:.1}%)   ",
                100.0 * done as f64 / total as f64
            )
        },
    )
    .map_err(|e| anyhow!("run_job: {e}"))?;
    eprintln!();
    let records = out.len() / manifest.record_size.max(1) as usize;

    // Compress the raw output records and submit them back to the fragment.
    let packed = compcol::vec::compress_to_vec::<compcol::xz::Xz>(&out)
        .map_err(|e| anyhow!("xz-compressing result: {e:?}"))?;
    submit_result(
        ctx,
        &pull.response_key,
        &packed,
        records as u64,
        t0.elapsed().as_secs_f64(),
    )?;
    Ok(true)
}

/// Submit the fragment's xz-compressed result via `Decrypt/Job:submit`.
fn submit_result(
    ctx: &RestContext,
    response_key: &str,
    body: &[u8],
    records: u64,
    secs: f64,
) -> Result<()> {
    let mut params: HashMap<String, Value> = HashMap::new();
    params.insert(
        "response_key".to_string(),
        Value::String(response_key.to_string()),
    );
    // :submit goes through the platform's standard upload (prepareCbCtx) flow.
    klbfw::upload(
        ctx,
        "Decrypt/Job:submit",
        "POST",
        params,
        Cursor::new(body.to_vec()),
        "application/x-xz",
        None,
    )
    .map_err(|e| anyhow!("Decrypt/Job:submit: {e}"))?;
    eprintln!(
        "[decryptd] submitted {records} record(s) ({} B xz) in {secs:.1}s",
        body.len()
    );
    Ok(())
}

fn main() -> Result<()> {
    run_worker(RunArgs::parse())
}

fn run_worker(args: RunArgs) -> Result<()> {
    std::fs::create_dir_all(&args.workdir)?;
    let ctx = RestContext::with_config(Config::new("https".to_string(), args.host.clone()))
        .with_debug(std::env::var("DECRYPTD_DEBUG").is_ok());
    eprintln!("[decryptd] host={}", args.host);

    loop {
        match run_once(&args, &ctx) {
            Ok(true) => {}
            Ok(false) => {
                if args.once {
                    eprintln!("[decryptd] no work; exiting (--once)");
                    return Ok(());
                }
                eprintln!("[decryptd] no work; sleeping {}s", args.idle_secs);
                std::thread::sleep(Duration::from_secs(args.idle_secs));
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
