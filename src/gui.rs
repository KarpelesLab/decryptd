//! System-tray front-end for the worker (Windows + Linux, behind the `gui`
//! feature).
//!
//! The worker pipeline is a set of background threads (see [`crate::run_worker`]).
//! A tray UI, however, must own the process's main thread — both GTK (Linux) and
//! the Win32 message pump (Windows) require it. So [`run_with_tray`] spawns the
//! worker on a thread and drives the tray loop here, polling a shared
//! [`crate::Status`] to keep the "Waiting / Running" menu line current.
//!
//! The menu is intentionally small: a disabled version line, a disabled status
//! line we refresh once a second, one disabled line per GPU in use, a
//! Pause/Resume toggle that parks/unparks the worker, a "Check for Updates" item
//! that kicks off a one-shot self-update, and a Quit item that ends the process.

use anyhow::{Context, Result, anyhow};
use tray_icon::menu::{Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

use crate::{RunArgs, Status};

const VERSION: &str = env!("CARGO_PKG_VERSION");
/// The app logo, decoded + scaled to the tray icon at startup.
const LOGO_PNG: &[u8] = include_bytes!("../assets/decryptd.png");

/// Spawn the worker pipeline on a background thread and run the tray on this
/// (main) thread. If the tray can't be created — e.g. a headless Linux box with
/// no display/StatusNotifier host — fall back to just running the worker.
pub fn run_with_tray(args: RunArgs, status: Status) -> Result<()> {
    // Snapshot the GPU list for the menu before `args` moves into the worker.
    let gpu_labels = gpu_labels(&args);

    let worker_status = status.clone();
    let worker = std::thread::spawn(move || {
        if let Err(e) = crate::run_worker(args, worker_status) {
            eprintln!("[decryptd] worker error: {e:#}");
        }
    });

    if let Err(e) = run_tray(status, gpu_labels) {
        eprintln!("[decryptd] tray unavailable: {e:#}; running headless");
        let _ = worker.join();
    }
    Ok(())
}

/// The GPUs decryptd will use, as menu-ready labels ("GPU#0: <name>"). Mirrors the
/// worker's own selection (`--gpus`, else all detected) so the tray shows exactly
/// what's in play. Best-effort: any probe failure collapses to a single note.
fn gpu_labels(args: &RunArgs) -> Vec<String> {
    let count = match crate::cuda::device_count() {
        Ok(n) if n > 0 => n,
        _ => return vec!["GPU: none detected".to_string()],
    };
    let ordinals = crate::select_gpus(&args.gpus, count).unwrap_or_else(|_| (0..count).collect());
    ordinals
        .iter()
        .map(|&o| {
            let name = crate::cuda::device_name(o).unwrap_or_else(|_| "unknown".to_string());
            format!("GPU#{o}: {name}")
        })
        .collect()
}

/// Decode the embedded logo and build a tray-sized RGBA icon.
fn load_icon() -> Result<Icon> {
    // Tray icons are small; downscale the 1254² source so we don't ship a few MB
    // of pixels to the platform's tray for something rendered at ~16–32 px.
    let img = image::load_from_memory(LOGO_PNG)
        .context("decoding logo png")?
        .resize_exact(64, 64, image::imageops::FilterType::Lanczos3)
        .to_rgba8();
    let (w, h) = img.dimensions();
    Icon::from_rgba(img.into_raw(), w, h).map_err(|e| anyhow!("building tray icon: {e}"))
}

/// The status line, refreshed each tick. Paused wins over running: while a paused
/// fragment sits parked between tiles it still counts as "active".
fn status_label(status: &Status) -> &'static str {
    if status.is_paused() {
        "Status: Paused"
    } else if status.is_running() {
        "Status: Running"
    } else {
        "Status: Waiting"
    }
}

/// The Pause/Resume toggle's label, refreshed each tick to match the flag.
fn pause_label(status: &Status) -> &'static str {
    if status.is_paused() {
        "Resume"
    } else {
        "Pause"
    }
}

/// The live menu bits the tick loop needs after build: the two items whose text
/// it refreshes, and the ids it routes clicks to. The `TrayIcon` is returned
/// separately (it must be kept alive for the tray to stay visible).
struct MenuHandles {
    status_item: MenuItem,
    pause_item: MenuItem,
    pause_id: MenuId,
    check_id: MenuId,
    quit_id: MenuId,
}

/// Build the tray icon and its context menu.
fn build_tray(status: &Status, gpu_labels: &[String]) -> Result<(TrayIcon, MenuHandles)> {
    let menu = Menu::new();
    let version = MenuItem::new(format!("decryptd v{VERSION}"), false, None);
    let status_item = MenuItem::new(status_label(status), false, None);
    let pause_item = MenuItem::new(pause_label(status), true, None);
    let check_updates = MenuItem::new("Check for Updates", true, None);
    let quit = MenuItem::new("Quit", true, None);
    menu.append(&version)?;
    menu.append(&status_item)?;
    // One disabled line per GPU decryptd is using.
    menu.append(&PredefinedMenuItem::separator())?;
    for label in gpu_labels {
        menu.append(&MenuItem::new(label, false, None))?;
    }
    menu.append(&PredefinedMenuItem::separator())?;
    menu.append(&pause_item)?;
    menu.append(&check_updates)?;
    menu.append(&quit)?;

    let tray = TrayIconBuilder::new()
        .with_tooltip(format!("decryptd v{VERSION}"))
        .with_icon(load_icon()?)
        .with_menu(Box::new(menu))
        .build()
        .map_err(|e| anyhow!("creating tray icon: {e}"))?;

    // The tray is live now — tell the user where to look for it.
    notify_running();

    let handles = MenuHandles {
        status_item,
        pause_id: pause_item.id().clone(),
        pause_item,
        check_id: check_updates.id().clone(),
        quit_id: quit.id().clone(),
    };
    Ok((tray, handles))
}

/// Pop a desktop notification announcing that the app is up, pointing the user
/// at the tray icon. Best-effort: with no notification service (a headless
/// session, no daemon) we just log and carry on — the tray is unaffected.
fn notify_running() {
    let res = notify_rust::Notification::new()
        .summary(&format!("decryptd v{VERSION}"))
        .body("decryptd is now running — right-click the tray icon for details.")
        .show();
    if let Err(e) = res {
        eprintln!("[decryptd] launch notification unavailable: {e}");
    }
}

/// Kick off a one-shot self-update from the tray. Reuses [`crate::build_updater`]
/// — the same signed updater the hourly background task uses — so it shares the
/// trust anchor and transport; `update()` fetches the latest signed manifest,
/// installs it if it's a genuinely newer build, and re-execs into it. Runs on
/// its own thread so a slow network never freezes the menu, and is a no-op (bar
/// a log line) when we're already current.
fn trigger_update_check() {
    std::thread::spawn(|| match crate::build_updater() {
        Ok(updater) => match updater.update() {
            Ok(true) => {} // installed; the process is being replaced
            Ok(false) => eprintln!("[decryptd] already up to date"),
            Err(e) => eprintln!("[decryptd] update check failed: {e}"),
        },
        Err(e) => eprintln!("[decryptd] updater unavailable: {e}"),
    });
}

/// Refresh the status + pause labels and drain pending menu clicks. Toggles pause,
/// kicks an update check, or exits on Quit. Called once per tick from each
/// platform's event loop.
fn tick(status: &Status, h: &MenuHandles) {
    h.status_item.set_text(status_label(status));
    h.pause_item.set_text(pause_label(status));
    while let Ok(event) = MenuEvent::receiver().try_recv() {
        if event.id == h.quit_id {
            std::process::exit(0);
        } else if event.id == h.pause_id {
            status.set_paused(!status.is_paused());
        } else if event.id == h.check_id {
            trigger_update_check();
        }
    }
}

#[cfg(target_os = "linux")]
fn run_tray(status: Status, gpu_labels: Vec<String>) -> Result<()> {
    gtk::init().map_err(|e| anyhow!("gtk init: {e}"))?;
    // Kept alive for the lifetime of `gtk::main()` below, which never returns.
    let (_tray, handles) = build_tray(&status, &gpu_labels)?;

    glib::timeout_add_seconds_local(1, move || {
        tick(&status, &handles);
        glib::ControlFlow::Continue
    });

    gtk::main();
    Ok(())
}

#[cfg(target_os = "windows")]
fn run_tray(status: Status, gpu_labels: Vec<String>) -> Result<()> {
    use std::time::Duration;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        DispatchMessageW, MSG, PM_REMOVE, PeekMessageW, TranslateMessage,
    };

    // Kept alive for the lifetime of the message loop below.
    let (_tray, handles) = build_tray(&status, &gpu_labels)?;

    let mut msg: MSG = unsafe { std::mem::zeroed() };
    loop {
        // Pump every pending message so the tray stays responsive...
        unsafe {
            while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
        // ...then refresh status / handle clicks and idle briefly.
        tick(&status, &handles);
        std::thread::sleep(Duration::from_millis(300));
    }
}
