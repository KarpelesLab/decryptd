//! Link the CUDA driver (libcuda) for decryptd's generic launch path. No nvcc is
//! used — decryptd only needs the NVIDIA driver (which every volunteer has).

use std::path::Path;

fn main() {
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
