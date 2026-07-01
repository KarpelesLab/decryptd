# decryptd

A GPU worker for **decrypt**. Run it on a machine with an NVIDIA GPU
and it quietly does distributed compute jobs in the background: it asks the
coordinator for a chunk of work, runs it on your GPU, sends the result back, and
repeats — forever, until you stop it. Run and forget.

## Requirements

- A **CUDA-capable NVIDIA GPU**.
- An **up-to-date NVIDIA driver**. No CUDA toolkit needed.
- Linux or Windows (64-bit).
- **Linux only:** the release binary includes the system-tray UI, so it needs
  GTK 3 and an AppIndicator library present — `libgtk-3`,
  `libayatana-appindicator3`, and `libxdo`. These ship with essentially every
  desktop install. On a **headless server**, install them (e.g.
  `apt install libgtk-3-0 libayatana-appindicator3-1 libxdo3`) or build without
  the tray (`cargo build --release`). Windows bundles everything in the `.exe`.

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
| `--jobs <N>` | `1` | How many chunks to run on the GPU at once. |
| `--workdir <DIR>` | `decryptd-data` | Where to keep the download cache and scratch files. |

Downloading the next chunk and uploading finished results always happen in the
background while the GPU works, so the card stays busy. `--jobs` only raises how
many run *on the GPU* simultaneously — most setups are fine with the default.

Run `decryptd --help` for the full list.

### System-tray mode

The released binaries run as a **system-tray app** on Windows and Linux. decryptd
sits in the tray with a right-click menu showing the version, the current status
(*Waiting* or *Running*), and a **Quit** entry. The worker runs in the
background exactly as above.

If no tray host is available (for example a Linux box with no desktop session),
decryptd logs a notice and falls back to running headless. Passing `--once` also
runs headless, with no tray.

To build a **console-only** binary with no tray (and no GUI dependencies), build
the default feature set — see [Building from source](#building-from-source).

## Building from source

You need a [Rust toolchain](https://rustup.rs) and a CUDA toolkit (only to link
against the driver library at build time — the binary still just needs the
driver to run):

```sh
cargo build --release                 # console-only, no GUI dependencies
cargo build --release --features gui   # system-tray app (what the releases ship)
```

The binary lands in `target/release/decryptd` (`.exe` on Windows). Building the
`gui` feature on Linux additionally needs the GTK 3 / AppIndicator / libxdo
development headers (`libgtk-3-dev libayatana-appindicator3-dev libxdo-dev`).

## License

Proprietary. See `Cargo.toml`.
