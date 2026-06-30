//! Producer side — publish a sweep job to the KarpelesLab platform via `klbfw`.
//!
//! A job is two opaque blobs plus a numeric range:
//!   1. `engine.zip` — the per-arch GPU cubins (what was historically "job.zip").
//!   2. `data.bin.xz` — the opaque, xz-compressed kernel input (bloom filters +
//!      config). Optional (a self-contained kernel may need none).
//!
//! Both are uploaded as `Decrypt\Data` blobs (idempotent, keyed by content hash), and
//! a `Decrypt\Job` is created over `[start, end)` referencing both. Workers later claim
//! fragments with `Decrypt\Job:pullOne` and submit results with `:submitFragment`.
//!
//! The join table `Decrypt_Job_Data` does not preserve order, so a worker tells the
//! engine zip from the data blob by magic bytes (`PK\x03\x04` vs xz `\xFD7zXZ`), not by
//! position — nothing here needs to encode a role.

use std::collections::HashMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use clap::Args;
use klbfw::{ApiKey, Config, RestContext};
use serde_json::{Value, json};

#[derive(Args)]
pub struct SubmitArgs {
    /// Engine archive: the per-arch GPU cubins (`*.sm<NN>.cubin`), zipped.
    #[arg(long, value_name = "ENGINE.ZIP")]
    engine: PathBuf,
    /// Opaque job data blob, xz-compressed (bloom filters + kernel config). Optional.
    #[arg(long, value_name = "DATA.BIN.XZ")]
    data: Option<PathBuf>,
    /// Range start (inclusive). Caller-defined units (seed / unix second / …).
    #[arg(long)]
    start: u64,
    /// Range end (exclusive) — matches Decrypt\\Job's half-open `[Range_Start, Range_End)`.
    #[arg(long)]
    end: u64,
    /// KarpelesLab API host.
    #[arg(long, env = "DECRYPTD_HOST", default_value = "www.atonline.com")]
    host: String,
    /// API key id (creating a job requires a trusted caller).
    #[arg(long, env = "DECRYPTD_API_KEY")]
    api_key: String,
    /// API key secret (base64 Ed25519 secret or 64-byte keypair).
    #[arg(long, env = "DECRYPTD_API_SECRET")]
    api_secret: String,
}

fn sha256_hex(b: &[u8]) -> String {
    purecrypto::hash::sha256(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

/// Upload a file as a `Decrypt\Data` blob (keyed by its content hash, so re-uploading
/// the same bytes is a no-op that returns the existing row). Returns its `Decrypt_Data__`.
fn upload_blob(ctx: &RestContext, path: &Path, mime: &str) -> Result<(String, usize)> {
    let bytes = std::fs::read(path).map_err(|e| anyhow!("reading {}: {e}", path.display()))?;
    let key = sha256_hex(&bytes);
    let n = bytes.len();
    let mut params: HashMap<String, Value> = HashMap::new();
    params.insert("key".to_string(), Value::String(key));
    // `klbfw::upload` runs the platform's prepare→PUT→complete flow and returns the
    // finalized row; `size` is injected from the reader automatically.
    let resp = klbfw::upload(
        ctx,
        "Decrypt/Data:upload",
        "POST",
        params,
        Cursor::new(bytes),
        mime,
        None,
    )
    .map_err(|e| anyhow!("Decrypt/Data:upload {}: {e}", path.display()))?;
    let uuid = resp.get_string("Decrypt_Data__").ok_or_else(|| {
        anyhow!(
            "upload of {} returned no Decrypt_Data__ ({:?})",
            path.display(),
            resp.raw()
        )
    })?;
    Ok((uuid, n))
}

pub fn run(a: SubmitArgs) -> Result<()> {
    if a.end <= a.start {
        bail!(
            "--end ({}) must be > --start ({}) [range is half-open)",
            a.end,
            a.start
        );
    }
    if !a.engine.exists() {
        bail!("engine archive not found: {}", a.engine.display());
    }
    let api_key = ApiKey::new(a.api_key.clone(), &a.api_secret)
        .map_err(|e| anyhow!("invalid --api-key/--api-secret: {e}"))?;
    let ctx = RestContext::with_config(Config::new("https".to_string(), a.host.clone()))
        .with_api_key(api_key);

    eprintln!("[submit] host={} range=[{}, {})", a.host, a.start, a.end);

    let (engine_uuid, en) = upload_blob(&ctx, &a.engine, "application/zip")?;
    eprintln!(
        "[submit] engine {} ({en} B) -> Decrypt/Data {engine_uuid}",
        a.engine.display()
    );

    let mut data_refs = vec![Value::String(engine_uuid)];
    if let Some(dp) = &a.data {
        if !dp.exists() {
            bail!("data blob not found: {}", dp.display());
        }
        let (data_uuid, dn) = upload_blob(&ctx, dp, "application/x-xz")?;
        eprintln!(
            "[submit] data   {} ({dn} B) -> Decrypt/Data {data_uuid}",
            dp.display()
        );
        data_refs.push(Value::String(data_uuid));
    }

    let resp = ctx
        .do_request(
            "Decrypt/Job",
            "POST",
            json!({
                "Range_Start": a.start,
                "Range_End": a.end,
                "Data": data_refs,
            }),
        )
        .map_err(|e| anyhow!("creating Decrypt/Job: {e}"))?;
    let job = resp
        .get_string("Decrypt_Job__")
        .ok_or_else(|| anyhow!("job creation returned no Decrypt_Job__ ({:?})", resp.raw()))?;

    // The job UUID on stdout (scriptable); everything else on stderr.
    println!("{job}");
    eprintln!(
        "[submit] created Decrypt/Job {job} over [{}, {}) with {} blob(s)",
        a.start,
        a.end,
        data_refs.len(),
    );
    eprintln!("[submit] workers: Decrypt/Job:pullOne?Decrypt_Job__={job}");
    Ok(())
}
