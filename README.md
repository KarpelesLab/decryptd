# decryptd

Generic volunteer GPU job runner for **decrypt**.

`decryptd` knows nothing about bloom filters, RNG, or BIP39 — it is a thin,
job-agnostic CUDA launcher. All job-specific logic lives in the GPU cubin and an
opaque data blob published to the KarpelesLab platform. A worker just:

1. claims a fragment of work with `Decrypt/Job:pullOne`,
2. downloads the job's blobs — `engine.zip` (the per-arch cubins + a
   `manifest.json`) and an optional `data.xz` — from the inline URLs the pull
   response carries,
3. reads the launch parameters from `manifest.json`, loads the cubin for the
   local GPU, and launches its kernel over the fragment range,
4. gathers the output records, compresses them (xz), and submits them back with
   `Decrypt/Job:submitFragment`.

`pullOne`/`submitFragment` and the `Decrypt/Data` blobs are anonymous-accessible,
so a worker needs **no credentials**. Because the kernel ABI is fixed (see
below), the same `decryptd` binary runs any job the platform hands it.

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

`decryptd` is a single worker command. Point it at the platform host and it loops
forever, claiming and running fragments:

```sh
decryptd                       # uses the default host www.atonline.com
decryptd --host www.atonline.com --once
```

| Flag | Env | Default | Description |
| --- | --- | --- | --- |
| `--host` | `DECRYPTD_HOST` | `www.atonline.com` | KarpelesLab platform host (anonymous — no key needed). |
| `--workdir` | `DECRYPTD_WORKDIR` | `decryptd-data` | Blob cache and scratch directory. |
| `--once` | | off | Claim a single fragment then exit. |
| `--idle-secs` | | `60` | Seconds to wait before retrying when there's no work. |

When `pullOne` reports no open work, the worker waits `--idle-secs` (one minute by
default) and tries again.

### Job layout

A job is published to the platform as two `Decrypt/Data` blobs referenced by a
`Decrypt/Job` over a half-open range `[Range_Start, Range_End)`:

- **`engine.zip`** — the per-arch GPU cubins (`*.sm<NN>.cubin`) plus a
  `manifest.json` carrying the launch parameters:

  ```json
  {
    "entry": "decrypt",
    "record_size": 28,
    "out_cap": 1048576,
    "block": 256,
    "tile": 16777216
  }
  ```

  Only `record_size` is required; the rest default as shown.
- **`data.xz`** *(optional)* — the opaque, xz-compressed kernel input.

The worker tells the two apart by filename (`.zip` vs `.xz`), falling back to the
file's magic bytes.

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

The record layout is chosen by `manifest.json`'s `record_size`; the kernel writes
`record_size`-byte records into `out` and bumps `out_count`.

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
