//! Build script: links the CUDA driver (libcuda) for decryptd's generic launch
//! path — no nvcc, only the NVIDIA driver every volunteer already has — and, on
//! Windows, embeds the application icon into the executable.

use std::path::Path;

fn main() {
    link_cuda();
    embed_windows_icon();
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
