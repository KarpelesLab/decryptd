# decryptd

A GPU worker for **decrypt**. Run it on a machine with an NVIDIA GPU
and it quietly does distributed compute jobs in the background: it asks the
coordinator for a chunk of work, runs it on your GPU, sends the result back, and
repeats — forever, until you stop it. Run and forget.

## ⚠️ Heat and power

decryptd keeps your GPU at **full load, continuously** — that's how it does the
work. A card pinned at 100% for hours pulls serious power and runs **hot**, and
sustained heat is what wears graphics hardware out.

- Make sure the machine has **good ventilation** and unobstructed airflow.
  Enclosed cases, laptops, and dust-clogged fans that cope with short bursts can
  overheat under a non-stop load.
- Keep an eye on temperature (`nvidia-smi -l 5` on Linux, or any GPU monitor). If
  the card runs hot, **stop decryptd** — `Ctrl-C`, or **Quit** from the tray — and
  improve cooling before running it again.
- Expect a higher **power bill** and a warmer room. On a laptop, run it plugged in
  rather than on battery.

If in doubt, stop it — the work is distributed, so one machine sitting out is
completely fine.

## Requirements

- A **CUDA-capable NVIDIA GPU**.
- An **up-to-date NVIDIA driver**. No CUDA toolkit needed.
- Linux or Windows (64-bit).
- **No GUI libraries to install.** The system tray loads whatever the desktop
  provides at runtime, so it just works on a desktop and is silently skipped on a
  headless server — the same binary runs either way.

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

### Run in a container

The included `Dockerfile` builds a small (~250 MB) image containing just the
worker. The GPU is supplied at run time by the
[NVIDIA container runtime](https://github.com/NVIDIA/nvidia-container-toolkit) —
no CUDA toolkit or GUI libraries go in the image.

```sh
docker build -t decryptd .
docker run -d --name decryptd --restart unless-stopped --gpus all \
  -v decryptd-data:/data decryptd
```

Mount a volume at `/data` to keep the worker id and download cache across
restarts. On **RunPod** (or any container host), push the image to a registry and
use it as the pod's image: the entrypoint launches the worker, so it starts
automatically and comes back after a restart — no systemd needed.

### Options

You normally don't need any of these.

| Option | Default | What it does |
| --- | --- | --- |
| `--once` | off | Do a single chunk of work, then exit (handy for testing). |
| `--jobs <N>` | `1` | How many chunks to run on the GPU at once, **per GPU**. |
| `--gpus <LIST>` | all | Which GPUs to use, e.g. `--gpus 0,2`. Default: every detected GPU, each with its own work queue. |
| `--idle-secs <N>` | `60` | How long to wait before re-checking when there's no work. |
| `--cache-max-gb <N>` | `20` | Cap the on-disk download cache; oldest blobs are evicted past this. |
| `--workdir <DIR>` | per-user data dir | Where to keep the download cache, worker id, and scratch. Defaults to a per-user folder (`%LOCALAPPDATA%\decryptd` / `~/.local/share/decryptd`). |

Downloading the next chunk and uploading finished results always happen in the
background while the GPU works, so the card stays busy. `--jobs` only raises how
many run *on the GPU* simultaneously — most setups are fine with the default.
With more than one GPU, `--jobs` applies to each, so `--jobs 3` on 2 GPUs runs 6
chunks in total.

Run `decryptd --help` for the full list.

### System-tray mode

The released binaries run as a **system-tray app** on Windows and Linux. Right-click
the tray icon for a menu showing the version, current status (*Waiting* /
*Running* / *Paused*), the GPU(s) in use, the live **speed** (tries per second,
1-minute average), a **Pause/Resume** toggle, **Check for Updates**, and **Quit**.
The worker runs in the background exactly as above.

The tray loads the desktop's toolkit at runtime, so there are **no GUI libraries
to install**. If no tray host is available — a headless server, or `--once` —
decryptd logs a notice and runs without a tray.

On Unix (with or without a tray), **`kill -USR1 <pid>`** toggles Pause/Resume — the
pause control for a headless worker, and scriptable on the desktop. The chosen
state is remembered on disk (a `paused` marker in the working directory), so a
paused worker stays paused across a restart — including an auto-update — instead of
silently resuming work.

## Building from source

You need a [Rust toolchain](https://rustup.rs) and a CUDA toolkit (only to link
against the driver library at build time — the binary still just needs the
driver to run):

```sh
cargo build --release                        # system-tray app (what the releases ship)
cargo build --release --no-default-features   # pure worker, no tray
```

The binary lands in `target/release/decryptd` (`.exe` on Windows). **No GUI system
packages are needed** either way: the tray backend (`ldtray`) links nothing at
build time and loads the platform toolkit at runtime.

## License

Proprietary. See `Cargo.toml`.
