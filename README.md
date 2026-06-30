# decryptd

A volunteer GPU worker for **decrypt**. Run it on a machine with an NVIDIA GPU
and it quietly does distributed compute jobs in the background: it asks the
coordinator for a chunk of work, runs it on your GPU, sends the result back, and
repeats — forever, until you stop it. Run and forget.

## Requirements

- A **CUDA-capable NVIDIA GPU**.
- An **up-to-date NVIDIA driver**. That's all — no CUDA toolkit, no extra
  runtime, nothing else to install.
- Linux or Windows (64-bit).

## Get it

Download the latest archive from the [Releases](../../releases) page and unpack
it:

- **Linux:** `decryptd-linux-x86_64.tar.gz` → `tar -xzf decryptd-linux-x86_64.tar.gz`
- **Windows:** `decryptd-windows-x86_64.zip` → extract it (right-click → Extract All)

Each archive contains a single `decryptd` executable.

## Run it

Just run it — no configuration needed:

```sh
./decryptd        # Linux
decryptd.exe      # Windows
```

It loops forever: claiming work, running it on the GPU, submitting results. When
there's no work available it waits a minute and checks again. Stop it any time
with `Ctrl-C`.

### Leaving it running

To keep it going after you log out:

```sh
# Linux — quick and dirty
nohup ./decryptd >decryptd.log 2>&1 &
```

For an always-on contributor, run it under a service manager (systemd on Linux,
a scheduled task / service on Windows) so it restarts on boot.

### Options

You normally don't need any of these.

| Option | Default | What it does |
| --- | --- | --- |
| `--once` | off | Do a single chunk of work, then exit (handy for testing). |
| `--idle-secs <N>` | `60` | How long to wait before re-checking when there's no work. |
| `--workdir <DIR>` | `decryptd-data` | Where to keep the download cache and scratch files. |

Run `decryptd --help` for the full list.

## Building from source

You need a [Rust toolchain](https://rustup.rs) and a CUDA toolkit (only to link
against the driver library at build time — the binary still just needs the
driver to run):

```sh
cargo build --release
```

The binary lands in `target/release/decryptd` (`.exe` on Windows).

## License

Proprietary. See `Cargo.toml`.
