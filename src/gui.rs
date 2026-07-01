//! System-tray front-end for the worker, via [`ldtray`].
//!
//! ldtray resolves every platform toolkit at *runtime* (libdbus/shell32/AppKit
//! through `libloading`), so this single binary runs everywhere: on a desktop it
//! shows a tray icon; on a headless server [`Tray::new`] simply returns an error
//! and we run the worker without a tray. No compile-time GUI linkage, no separate
//! console build.
//!
//! ldtray's model is event-driven: `Tray::run` owns the main thread and delivers
//! click [`Event`]s, while a [`TrayHandle`] (cloneable, `Send`) pushes updates back
//! in — a new [`Menu`], a [`Notification`], etc. So the worker runs on a background
//! thread, a second thread rebuilds the menu once a second to track the worker's
//! status, and menu clicks are handled in the run callback.
//!
//! The menu: a disabled version line, a disabled status line (Waiting / Running /
//! Paused, or the reason it's idle), one disabled line per GPU in use, a
//! Pause/Resume toggle, a "Check for Updates" item, and Quit.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use ldtray::{Event, Icon, Menu, MenuId, MenuItem, Notification, Tray, TrayConfig, TrayHandle};

use crate::nvml::Nvml;
use crate::{RunArgs, Status};

/// A GPU decryptd is using, resolved once at startup. `pci_bus_id` keys the live
/// NVML temperature/power lookup.
#[derive(Clone)]
struct GpuInfo {
    ordinal: i32,
    name: String,
    pci_bus_id: Option<String>,
}

const VERSION: &str = env!("CARGO_PKG_VERSION");
/// The app logo, decoded + scaled to the tray icon at startup.
const LOGO_PNG: &[u8] = include_bytes!("../assets/decryptd.png");

// Menu item ids echoed back in `Event::Menu`. Disabled/label-only rows use
// `ID_INERT` since they never emit an event.
const ID_INERT: u32 = 0;
const ID_PAUSE: u32 = 1;
const ID_CHECK: u32 = 2;
const ID_QUIT: u32 = 3;

/// Spawn the worker and run the tray on this (main) thread. If the tray can't be
/// created — a headless box with no display/notification host — fall back to just
/// running the worker.
pub fn run_with_tray(args: RunArgs, status: Status) -> Result<()> {
    let gpus: Arc<Vec<GpuInfo>> = Arc::new(gpu_infos(&args));
    // NVML (driver-provided) for live temp/power; None → the tray just omits it.
    let nvml: Option<Arc<Nvml>> = Nvml::load().map(Arc::new);

    let icon = match load_icon() {
        Ok(icon) => icon,
        Err(e) => {
            eprintln!("[decryptd] tray icon unavailable: {e}; running headless");
            return crate::run_worker(args, status);
        }
    };
    let config = TrayConfig::new(icon)
        .tooltip(format!("decryptd v{VERSION}"))
        .menu(build_menu(&status, &gpus, nvml.as_deref()));
    let tray = match Tray::new(config) {
        Ok(tray) => tray,
        Err(e) => {
            eprintln!("[decryptd] tray unavailable: {e}; running headless");
            return crate::run_worker(args, status);
        }
    };
    let handle = tray.handle();

    // Worker on a background thread; a fatal stop is surfaced in the status line
    // and as a desktop notification (the GUI subsystem has no console).
    {
        let (status, handle) = (status.clone(), handle.clone());
        std::thread::spawn(move || {
            if let Err(e) = crate::run_worker(args, status.clone()) {
                let msg = format!("{e:#}");
                eprintln!("[decryptd] worker error: {msg}");
                status.set_note(format!("stopped: {msg}"));
                let _ = handle.notify(Notification::new(
                    format!("decryptd v{VERSION} stopped"),
                    msg,
                ));
            }
        });
    }

    // Rebuild the menu once a second so status, speed, and live GPU telemetry
    // track the worker. NVML reads are sub-millisecond, so this cadence is cheap.
    {
        let (status, handle, gpus, nvml) =
            (status.clone(), handle.clone(), gpus.clone(), nvml.clone());
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_secs(1));
                if handle
                    .set_menu(build_menu(&status, &gpus, nvml.as_deref()))
                    .is_err()
                {
                    break; // tray loop gone
                }
            }
        });
    }

    // The tray is live — point the user at it.
    let _ = handle.notify(Notification::new(
        format!("decryptd v{VERSION}"),
        "running — right-click the tray icon for details",
    ));

    // Drive the tray event loop on the main thread; it returns only on error
    // (Quit calls `process::exit`).
    let cb = handle.clone();
    tray.run(move |event| {
        if let Event::Menu(MenuId(id)) = event {
            match id {
                ID_QUIT => std::process::exit(0),
                ID_PAUSE => {
                    status.set_paused(!status.is_paused());
                    // Flip the label immediately rather than waiting for the tick.
                    let _ = cb.set_menu(build_menu(&status, &gpus, nvml.as_deref()));
                }
                ID_CHECK => trigger_update_check(cb.clone()),
                _ => {}
            }
        }
    })
    .map_err(|e| anyhow!("tray loop: {e}"))?;
    Ok(())
}

/// The GPUs decryptd will use, resolved once at startup. Mirrors the worker's own
/// selection (`--gpus`, else all detected). Best-effort; empty when none detected.
fn gpu_infos(args: &RunArgs) -> Vec<GpuInfo> {
    let count = match crate::cuda::device_count() {
        Ok(n) if n > 0 => n,
        _ => return Vec::new(),
    };
    let ordinals = crate::select_gpus(&args.gpus, count).unwrap_or_else(|_| (0..count).collect());
    ordinals
        .iter()
        .map(|&ordinal| GpuInfo {
            ordinal,
            name: crate::cuda::device_name(ordinal).unwrap_or_else(|_| "unknown".to_string()),
            pci_bus_id: crate::cuda::pci_bus_id(ordinal),
        })
        .collect()
}

/// One GPU's menu line: `GPU#0: NVIDIA GeForce RTX 3080 — 72 °C, 220 W`, with the
/// telemetry appended only when NVML can supply it.
fn gpu_line(info: &GpuInfo, nvml: Option<&Nvml>) -> String {
    let mut line = format!("GPU#{}: {}", info.ordinal, info.name);
    if let (Some(nvml), Some(pci)) = (nvml, info.pci_bus_id.as_deref()) {
        let t = nvml.telemetry(pci);
        let mut extra: Vec<String> = Vec::new();
        if let Some(c) = t.temp_c {
            extra.push(format!("{c} °C"));
        }
        if let Some(w) = t.power_w {
            extra.push(format!("{w:.0} W"));
        }
        if !extra.is_empty() {
            line = format!("{line} — {}", extra.join(", "));
        }
    }
    line
}

/// Decode the embedded logo into the small RGBA image an ldtray [`Icon`] wants.
fn load_icon() -> Result<Icon> {
    let img = image::load_from_memory(LOGO_PNG)
        .context("decoding logo png")?
        .resize_exact(32, 32, image::imageops::FilterType::Lanczos3)
        .to_rgba8();
    let (w, h) = img.dimensions();
    Icon::from_rgba(w, h, img.into_raw()).map_err(|e| anyhow!("building tray icon: {e}"))
}

/// The status line. Paused wins over running (a paused fragment parked between
/// tiles still counts as "active"); when idle, the worker's note (why it's waiting)
/// is shown so a silent stall is legible.
fn status_label(status: &Status) -> String {
    if status.is_paused() {
        return "Status: Paused".to_string();
    }
    if status.is_running() {
        return "Status: Running".to_string();
    }
    let note = status.note();
    if note.is_empty() {
        return "Status: Waiting".to_string();
    }
    // Keep the menu from ballooning on a long error message.
    let note = if note.chars().count() > 60 {
        format!("{}…", note.chars().take(60).collect::<String>())
    } else {
        note
    };
    format!("Status: {note}")
}

/// Format an items/second rate with an SI-ish suffix — GPU rates run into the
/// billions, so the raw number is unreadable.
fn fmt_rate(r: f64) -> String {
    if r >= 1e12 {
        format!("{:.2} T/s", r / 1e12)
    } else if r >= 1e9 {
        format!("{:.2} G/s", r / 1e9)
    } else if r >= 1e6 {
        format!("{:.1} M/s", r / 1e6)
    } else if r >= 1e3 {
        format!("{:.1} k/s", r / 1e3)
    } else {
        format!("{r:.0}/s")
    }
}

/// The Pause/Resume toggle's label, matching the current flag.
fn pause_label(status: &Status) -> &'static str {
    if status.is_paused() {
        "Resume"
    } else {
        "Pause"
    }
}

/// Build the context menu for the current worker state. ldtray menus are
/// immutable snapshots, so the refresher re-renders this and pushes it via
/// [`TrayHandle::set_menu`] whenever the status changes.
fn build_menu(status: &Status, gpus: &[GpuInfo], nvml: Option<&Nvml>) -> Menu {
    let mut menu = Menu::new()
        .item(MenuItem::button(ID_INERT, format!("decryptd v{VERSION}")).enabled(false))
        .item(MenuItem::button(ID_INERT, status_label(status)).enabled(false));
    // Live throughput (1-minute average), shown once there's something to report.
    if let Some(rate) = status.tries_per_sec() {
        menu = menu.item(
            MenuItem::button(ID_INERT, format!("Speed: {} (1m avg)", fmt_rate(rate)))
                .enabled(false),
        );
    }
    menu = menu.item(MenuItem::separator());
    // One disabled line per GPU, with live temperature / power when NVML is there.
    if gpus.is_empty() {
        menu = menu.item(MenuItem::button(ID_INERT, "GPU: none detected").enabled(false));
    }
    for info in gpus {
        menu = menu.item(MenuItem::button(ID_INERT, gpu_line(info, nvml)).enabled(false));
    }
    menu.item(MenuItem::separator())
        .item(MenuItem::button(ID_PAUSE, pause_label(status)))
        .item(MenuItem::button(ID_CHECK, "Check for Updates"))
        .item(MenuItem::button(ID_QUIT, "Quit"))
}

/// Run a one-shot self-update, reporting the outcome via desktop notification.
/// This is the only feedback on the (console-less) GUI build, so "Check for
/// Updates" must never fail silently: the user always sees up-to-date, updating,
/// or the actual error. Runs on its own thread so a slow network never freezes the
/// menu; `update()` re-checks, installs, and restarts into the new build.
fn trigger_update_check(handle: TrayHandle) {
    std::thread::spawn(move || {
        let notify = |body: String| {
            let _ = handle.notify(Notification::new(
                format!("decryptd v{VERSION} — updates"),
                body,
            ));
        };
        let updater = match crate::build_updater() {
            Ok(updater) => updater,
            Err(e) => {
                notify(format!("Updater unavailable: {e}"));
                return;
            }
        };
        match updater.check() {
            Ok(Some(available)) => {
                let v = available.version().to_string();
                notify(format!("Updating to v{v} — decryptd will restart."));
                if let Err(e) = updater.update() {
                    notify(format!("Update to v{v} failed: {e}"));
                }
            }
            Ok(None) => notify(format!("decryptd is up to date (v{VERSION}).")),
            Err(e) => notify(format!("Update check failed: {e}")),
        }
    });
}
