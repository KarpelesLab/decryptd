//! Build script: links the CUDA driver (libcuda) for decryptd's generic launch
//! path — no nvcc, only the NVIDIA driver every volunteer already has — and, on
//! Windows, embeds the application icon into the executable.

use std::path::Path;
use std::process::Command;

fn main() {
    link_cuda();
    embed_windows_icon();
    stamp_build_identity();
}

/// Capture this build's git identity (short hash + commit time) as compile-time
/// env vars for the rsupd updater's "is this a newer build of the same version?"
/// check (see `build_updater` in `main.rs`). Empty when built outside a git
/// checkout (e.g. a crates.io tarball); the updater then falls back to a plain
/// semver comparison.
fn stamp_build_identity() {
    fn git(args: &[&str]) -> Option<String> {
        let out = Command::new("git").args(args).output().ok()?;
        if !out.status.success() {
            return None;
        }
        let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
        (!s.is_empty()).then_some(s)
    }

    let git_tag = git(&["rev-parse", "--short=7", "HEAD"]).unwrap_or_default();
    let build_unix = git(&["log", "-1", "--format=%ct", "HEAD"]).unwrap_or_default();
    println!("cargo:rustc-env=RSUPD_GIT_TAG={git_tag}");
    println!("cargo:rustc-env=RSUPD_BUILD_UNIX={build_unix}");

    // Re-stamp when the checked-out commit moves.
    println!("cargo:rerun-if-changed=.git/HEAD");
    if let Ok(head) = std::fs::read_to_string(".git/HEAD")
        && let Some(reference) = head.strip_prefix("ref:")
    {
        // Only the first line, and only a clean ref path — an embedded newline
        // could otherwise inject extra `cargo:` directives.
        let reference = reference.lines().next().unwrap_or("").trim();
        if !reference.is_empty()
            && !reference
                .chars()
                .any(|c| c.is_whitespace() || c.is_control())
        {
            println!("cargo:rerun-if-changed=.git/{reference}");
        }
    }
}

/// Embed the application icon and version-info resource into `decryptd.exe` so it
/// shows up in Explorer, the taskbar, the tray, and the file Properties dialog.
/// Compiled only when the build host is Windows (where `winresource` and rc.exe
/// exist); a no-op everywhere else, including when cross-compiling a Windows
/// target from Linux.
#[cfg(windows)]
fn embed_windows_icon() {
    println!("cargo:rerun-if-changed=assets/decryptd.ico");
    // The VERSIONINFO fields derive from Cargo metadata. A build script that emits
    // any `rerun-if-*` directive (link_cuda does) otherwise stops rerunning on
    // Cargo.toml edits, which would leave a stale version baked into the exe — so
    // re-run whenever any of these change.
    let env = |k: &str| std::env::var(k).unwrap_or_default();
    for k in [
        "CARGO_PKG_VERSION",
        "CARGO_PKG_NAME",
        "CARGO_PKG_AUTHORS",
        "CARGO_PKG_DESCRIPTION",
        "CARGO_PKG_REPOSITORY",
    ] {
        println!("cargo:rerun-if-env-changed={k}");
    }

    let name = env("CARGO_PKG_NAME");
    let description = env("CARGO_PKG_DESCRIPTION");
    let repository = env("CARGO_PKG_REPOSITORY");
    // CARGO_PKG_AUTHORS is colon-separated for multiple authors.
    let authors = env("CARGO_PKG_AUTHORS").replace(':', ", ");

    let mut res = winresource::WindowsResource::new();
    res.set_icon("assets/decryptd.ico");
    // winresource already fills FileVersion / ProductVersion / ProductName and a
    // FileDescription (defaulted to the crate name); add the rest of the standard
    // VERSIONINFO strings.
    if !description.is_empty() {
        res.set("FileDescription", &description);
    }
    if !authors.is_empty() {
        res.set("CompanyName", &authors);
        res.set("LegalCopyright", &format!("Copyright © {authors}"));
    }
    if !name.is_empty() {
        res.set("InternalName", &name);
        res.set("OriginalFilename", &format!("{name}.exe"));
    }
    if !repository.is_empty() {
        res.set("Comments", &repository);
    }

    if let Err(e) = res.compile() {
        println!("cargo:warning=failed to embed Windows resources: {e}");
    }
}

#[cfg(not(windows))]
fn embed_windows_icon() {}

fn link_cuda() {
    // Rerun when the relevant env vars change so CI/local installs are picked up.
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // Roots from env: CUDA_PATH is exported by the CI toolkit installer; CUDA_HOME
    // is the common local convention.
    let env_roots: Vec<String> = ["CUDA_PATH", "CUDA_HOME"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .collect();

    let mut dirs: Vec<String> = Vec::new();

    if target_os == "windows" {
        // The MSVC import library cuda.lib lives under lib\x64 of the toolkit.
        for root in &env_roots {
            dirs.push(format!("{root}\\lib\\x64"));
            dirs.push(format!("{root}\\lib"));
        }
    } else {
        // The real libcuda.so.1 is the installed driver; the toolkit also ships a stub.
        dirs.push("/usr/lib64".to_string());
        dirs.push("/usr/lib/x86_64-linux-gnu".to_string());
        dirs.push("/usr/lib".to_string());
        let mut roots = vec!["/opt/cuda".to_string(), "/usr/local/cuda".to_string()];
        roots.extend(env_roots.iter().cloned());
        for root in roots {
            dirs.push(format!("{root}/lib64"));
            dirs.push(format!("{root}/lib64/stubs"));
            dirs.push(format!("{root}/targets/x86_64-linux/lib"));
            dirs.push(format!("{root}/targets/x86_64-linux/lib/stubs"));
        }
    }

    for d in dirs {
        if Path::new(&d).exists() {
            println!("cargo:rustc-link-search=native={d}");
        }
    }
    println!("cargo:rustc-link-lib=dylib=cuda");
}
