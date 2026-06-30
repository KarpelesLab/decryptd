# decryptd

Generic volunteer GPU job runner for **decrypt**.

`decryptd` knows nothing about bloom filters, RNG, or BIP39 — it is a thin,
job-agnostic CUDA launcher. All job-specific logic lives in the GPU cubin and an
opaque data blob produced by the coordinator. A worker just:

1. asks the coordinator for a chunk,
2. downloads the cubin(s) (`job.zip`) and the opaque data blob (`data.xz`),
3. loads the cubin for the local GPU and launches its kernel over the range,
4. gathers the output records, compresses them (xz), and uploads them.

Because the kernel ABI is fixed (see below), the same `decryptd` binary runs any
job the coordinator hands it.

## Requirements

- **To run:** only the **NVIDIA driver** (`libcuda`). No CUDA toolkit, no `nvcc`,
  no runtime libraries — anything a volunteer with a working GPU already has.
- **To build:** a Rust toolchain (edition 2024, see `rust-version` in
  `Cargo.toml`) **and** a CUDA toolkit, which provides the link-time driver
  library (`libcuda.so` stub on Linux / `cuda.lib` on Windows). The toolkit is
  needed only to satisfy the linker — the binary loads the real driver at
  runtime. `build.rs` locates it automatically from common install paths and the
  `CUDA_PATH` / `CUDA_HOME` environment variables.

## Building

```sh
cargo build --release
```

The binary is written to `target/release/decryptd` (`.exe` on Windows).

Prebuilt `linux-x86_64` and `windows-x86_64` binaries are attached to each
`v#.#.#` GitHub release.

## Usage

`decryptd` has two subcommands: `run` (the volunteer worker) and `submit` (the
producer that publishes a job).

### Worker — `decryptd run`

Pulls chunks from a coordinator, runs the kernel, and uploads results, looping
forever until stopped.

```sh
decryptd run --server https://sweep.example.com
```

| Flag | Env | Default | Description |
| --- | --- | --- | --- |
| `--server` | `DECRYPTD_SERVER` | — | Coordinator base URL (required). |
| `--workdir` | `DECRYPTD_WORKDIR` | `decryptd-data` | Worker id, artifact cache, scratch. |
| `--token` | `DECRYPTD_TOKEN` | — | Optional bearer token for the coordinator. |
| `--once` | | off | Run a single chunk then exit. |
| `--idle-secs` | | `15` | Backoff base when the coordinator has no work. |

### Producer — `decryptd submit`

Publishes a job to the KarpelesLab `Decrypt/*` API: uploads `engine.zip` (the
per-arch cubins) and an optional `data.bin.xz` (the opaque kernel input) as
`Decrypt\Data` blobs, then creates a `Decrypt\Job` over the half-open range
`[start, end)` referencing both.

```sh
decryptd submit \
  --engine engine.zip \
  --data data.bin.xz \
  --start 0 --end 1000000000
```

| Flag | Env | Default | Description |
| --- | --- | --- | --- |
| `--engine` | | — | Engine archive: per-arch `*.sm<NN>.cubin`, zipped (required). |
| `--data` | | — | Opaque xz-compressed job data blob (optional). |
| `--start` | | — | Range start, inclusive (required). |
| `--end` | | — | Range end, exclusive (required). |
| `--host` | `DECRYPTD_HOST` | `www.atonline.com` | KarpelesLab API host. |
| `--api-key` | `DECRYPTD_API_KEY` | — | API key id (job creation needs a trusted caller). |
| `--api-secret` | `DECRYPTD_API_SECRET` | — | API key secret (base64 Ed25519 secret or 64-byte keypair). |

## Kernel ABI

Every cubin entry point a `decryptd` job ships must implement this fixed
signature (default entry name: `decrypt`):

```c
extern "C" __global__ void decrypt(
    unsigned long long start,    // first work-item index
    unsigned long long count,    // items in this launch
    const unsigned char* data,   // opaque job data blob (device)
    unsigned long long data_len,
    unsigned char* out,          // output record buffer (device)
    unsigned int* out_count,     // atomically-incremented record counter
    unsigned int out_cap);       // capacity in records
```

The coordinator chooses the record layout via the assignment's `record_size`;
the kernel writes `record_size`-byte records into `out` and bumps `out_count`.

## CI / releases

Three GitHub Actions workflows are defined under `.github/workflows/`:

- **`ci.yml`** — `rustfmt`, `clippy` (`-D warnings`), and `rustdoc`
  (`-D warnings`) on every push and pull request. These only analyze/document
  and never link the final binary, so they don't need the CUDA toolkit.
- **`build.yml`** — a reusable matrix that builds release binaries for
  `linux-x86_64` and `windows-x86_64`, installing the CUDA toolkit for the
  link step and uploading each binary as an artifact.
- **`release.yml`** — on a `v#.#.#` tag, reuses `build.yml` and attaches the
  built binaries to the GitHub release.

Cut a release with:

```sh
git tag v0.1.0
git push --tags
```

## License

Proprietary. See `Cargo.toml`.
