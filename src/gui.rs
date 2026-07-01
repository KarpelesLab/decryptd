//! System-tray front-end for the worker (Windows + Linux, behind the `gui`
//! feature).
//!
//! The worker pipeline is a set of background threads (see [`crate::run_worker`]).
//! A tray UI, however, must own the process's main thread — both GTK (Linux) and
//! the Win32 message pump (Windows) require it. So [`run_with_tray`] spawns the
//! worker on a thread and drives the tray loop here, polling a shared
//! [`crate::Status`] to keep the "Waiting / Running" menu line current.
//!
//! The menu is intentionally tiny: a disabled version line, a disabled status
//! line we refresh once a second, a "Check for Updates" item that kicks off a
//! one-shot self-update, and a Quit item that ends the process.

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
    let worker_status = status.clone();
    let worker = std::thread::spawn(move || {
        if let Err(e) = crate::run_worker(args, worker_status) {
            eprintln!("[decryptd] worker error: {e:#}");
        }
    });

    if let Err(e) = run_tray(status) {
        eprintln!("[decryptd] tray unavailable: {e:#}; running headless");
        let _ = worker.join();
    }
    Ok(())
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

/// The menu line we keep up to date as the worker switches between idle and busy.
fn status_label(status: &Status) -> &'static str {
    if status.is_running() {
        "Status: Running"
    } else {
        "Status: Waiting"
    }
}

/// Build the tray icon and its context menu. Returns the live `TrayIcon` (which
/// must be kept alive for the tray to stay visible), the status menu item (so
/// the loop can refresh its text), and the "Check for Updates" and Quit item ids.
fn build_tray(status: &Status) -> Result<(TrayIcon, MenuItem, MenuId, MenuId)> {
    let menu = Menu::new();
    let version = MenuItem::new(format!("decryptd v{VERSION}"), false, None);
    let status_item = MenuItem::new(status_label(status), false, None);
    let check_updates = MenuItem::new("Check for Updates", true, None);
    let quit = MenuItem::new("Quit", true, None);
    menu.append(&version)?;
    menu.append(&status_item)?;
    menu.append(&PredefinedMenuItem::separator())?;
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

    Ok((
        tray,
        status_item,
        check_updates.id().clone(),
        quit.id().clone(),
    ))
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

/// Refresh the status line and drain pending menu clicks. Exits the process when
/// Quit is chosen. Called once per tick from each platform's event loop.
fn tick(status: &Status, status_item: &MenuItem, check_id: &MenuId, quit_id: &MenuId) {
    status_item.set_text(status_label(status));
    while let Ok(event) = MenuEvent::receiver().try_recv() {
        if event.id == *quit_id {
            std::process::exit(0);
        } else if event.id == *check_id {
            trigger_update_check();
        }
    }
}

#[cfg(target_os = "linux")]
fn run_tray(status: Status) -> Result<()> {
    gtk::init().map_err(|e| anyhow!("gtk init: {e}"))?;
    // Kept alive for the lifetime of `gtk::main()` below, which never returns.
    let (_tray, status_item, check_id, quit_id) = build_tray(&status)?;

    glib::timeout_add_seconds_local(1, move || {
        tick(&status, &status_item, &check_id, &quit_id);
        glib::ControlFlow::Continue
    });

    gtk::main();
    Ok(())
}

#[cfg(target_os = "windows")]
fn run_tray(status: Status) -> Result<()> {
    use std::time::Duration;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        DispatchMessageW, MSG, PM_REMOVE, PeekMessageW, TranslateMessage,
    };

    // Kept alive for the lifetime of the message loop below.
    let (_tray, status_item, check_id, quit_id) = build_tray(&status)?;

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
        tick(&status, &status_item, &check_id, &quit_id);
        std::thread::sleep(Duration::from_millis(300));
    }
}
