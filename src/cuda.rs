//! Minimal CUDA Driver-API wrapper for the generic launch path. decryptd knows
//! nothing about the kernel's job — it uploads an opaque data blob, launches a
//! kernel with the fixed ABI below over a range, and reads back the output records.
//!
//! Kernel ABI v4 (the contract every current decryptd cubin implements; the full
//! spec is the publisher's `api.md`):
//! ```c
//! extern "C" __global__ void <entry>(
//!     unsigned long long* start,   // in/out: the work cell, 2 + nlimbs x u64
//!     unsigned long long count,    // items in this launch
//!     const unsigned char* data,   // the opaque job data blob (device)
//!     unsigned long long data_len,
//!     unsigned char* out,          // output byte-stream buffer (device)
//!     unsigned int* out_count,     // atomic BYTE cursor
//!     unsigned int out_cap);       // capacity in BYTES
//! ```
//! v4's only change from v3 is the work-item index: it is no longer fixed at 64
//! bits but a little-endian sequence of `nlimbs` 64-bit limbs, and the `start` cell
//! self-describes its width so a cubin built for a different one rejects the job
//! instead of computing garbage. Some keyspaces need it — Coldcard Mk4 is 89 bits,
//! and a 10-character brute over 95 glyphs is 3.2x `u64::MAX`.
//!
//! Both older ABIs are still supported for jobs already dispatched against them
//! (api.md §6): v3 (a 2-word cell, 64-bit index) and pre-v3 (`start` passed by
//! value, `out_count`/`out_cap` counted in fixed-size records). The manifest's
//! `format` field selects between them; see [`Abi`].

use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;
use std::time::{Duration, Instant};

type CuResult = i32;
type CuDevice = i32;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuDeviceptr = u64;

#[allow(non_snake_case)]
unsafe extern "C" {
    fn cuInit(flags: u32) -> CuResult;
    fn cuDeviceGetCount(count: *mut i32) -> CuResult;
    fn cuDeviceGet(device: *mut CuDevice, ordinal: i32) -> CuResult;
    fn cuDeviceGetAttribute(pi: *mut i32, attrib: i32, dev: CuDevice) -> CuResult;
    fn cuDeviceGetName(name: *mut c_char, len: i32, dev: CuDevice) -> CuResult;
    // Used only by the GUI's NVML telemetry, to map a CUDA ordinal to its physical
    // GPU (NVML doesn't honor CUDA_VISIBLE_DEVICES; the PCI id is the shared key).
    #[cfg(all(feature = "gui", any(target_os = "linux", target_os = "windows")))]
    fn cuDeviceGetPCIBusId(pci_bus_id: *mut c_char, len: i32, dev: CuDevice) -> CuResult;
    fn cuCtxCreate_v2(pctx: *mut CuContext, flags: u32, dev: CuDevice) -> CuResult;
    fn cuCtxDestroy_v2(ctx: CuContext) -> CuResult;
    fn cuModuleLoadData(module: *mut CuModule, image: *const c_void) -> CuResult;
    fn cuModuleUnload(module: CuModule) -> CuResult;
    fn cuModuleGetFunction(
        func: *mut CuFunction,
        module: CuModule,
        name: *const c_char,
    ) -> CuResult;
    fn cuMemAlloc_v2(dptr: *mut CuDeviceptr, bytes: usize) -> CuResult;
    fn cuMemFree_v2(dptr: CuDeviceptr) -> CuResult;
    fn cuMemcpyHtoD_v2(dst: CuDeviceptr, src: *const c_void, bytes: usize) -> CuResult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CuDeviceptr, bytes: usize) -> CuResult;
    fn cuMemsetD8_v2(dst: CuDeviceptr, uc: u8, n: usize) -> CuResult;
    fn cuLaunchKernel(
        f: CuFunction,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        shmem: u32,
        stream: *mut c_void,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CuResult;
    fn cuCtxSynchronize() -> CuResult;
    fn cuGetErrorString(err: CuResult, pstr: *mut *const c_char) -> CuResult;
}

const CU_DEV_ATTR_CC_MAJOR: i32 = 75;
const CU_DEV_ATTR_CC_MINOR: i32 = 76;

fn check(r: CuResult, what: &str) -> Result<(), String> {
    if r == 0 {
        return Ok(());
    }
    let mut s: *const c_char = ptr::null();
    let msg = unsafe {
        if cuGetErrorString(r, &mut s) == 0 && !s.is_null() {
            CStr::from_ptr(s).to_string_lossy().into_owned()
        } else {
            format!("CUDA error {r}")
        }
    };
    Err(format!("{what}: {msg}"))
}

/// A device allocation, freed on drop.
pub struct DeviceBuf {
    ptr: CuDeviceptr,
    len: usize,
}
impl DeviceBuf {
    fn alloc(len: usize) -> Result<DeviceBuf, String> {
        let mut p: CuDeviceptr = 0;
        check(unsafe { cuMemAlloc_v2(&mut p, len.max(1)) }, "cuMemAlloc")?;
        Ok(DeviceBuf { ptr: p, len })
    }
    fn from_slice(data: &[u8]) -> Result<DeviceBuf, String> {
        let b = DeviceBuf::alloc(data.len())?;
        if !data.is_empty() {
            check(
                unsafe { cuMemcpyHtoD_v2(b.ptr, data.as_ptr() as *const c_void, data.len()) },
                "cuMemcpyHtoD",
            )?;
        }
        Ok(b)
    }
    fn htod(&self, src: &[u8]) -> Result<(), String> {
        check(
            unsafe {
                cuMemcpyHtoD_v2(
                    self.ptr,
                    src.as_ptr() as *const c_void,
                    src.len().min(self.len),
                )
            },
            "cuMemcpyHtoD",
        )
    }
    fn memset0(&self) -> Result<(), String> {
        check(
            unsafe { cuMemsetD8_v2(self.ptr, 0, self.len) },
            "cuMemsetD8",
        )
    }
    fn dtoh(&self, dst: &mut [u8]) -> Result<(), String> {
        check(
            unsafe { cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut c_void, self.ptr, dst.len()) },
            "cuMemcpyDtoH",
        )
    }
}
impl Drop for DeviceBuf {
    fn drop(&mut self) {
        unsafe { cuMemFree_v2(self.ptr) };
    }
}

/// Which kernel ABI a job's cubins implement, selected by the manifest's `format`
/// (api.md §5). They differ in how `start` is passed, what `out_count`/`out_cap`
/// count, and how the output buffer is framed — see the module docs.
#[derive(Clone, Copy, Debug)]
pub enum Abi {
    /// Pre-v3 (`format` absent or `< 3`): `start` by value, `out_count`/`out_cap` in
    /// records, output a flat array of fixed-size records. Kept so a worker never
    /// chokes on jobs dispatched before v3 (api.md §6).
    Legacy { record_size: u32 },
    /// v3 (`format == 3`): `start` points at a `[base, resume]` cell,
    /// `out_count`/`out_cap` are byte counts, output is a self-delimiting stream of
    /// `uleb128(len) ‖ payload` records. decryptd never parses a payload — it only
    /// walks the length prefixes to find where the stream ends.
    V3,
    /// v4 (`format >= 4`): v3 with a self-describing work cell —
    /// `[base_0, resume, nlimbs, base_1 … base_{nlimbs-1}]` — so the work-item index
    /// can be wider than 64 bits. `index_words` is the manifest's `index_words`
    /// (`nlimbs`), which must equal the width the cubin was built for or it rejects
    /// the job outright.
    V4 { index_words: u32 },
}

/// Upper bound on the manifest's `index_words`. Nothing needs more than two limbs
/// today (the widest documented index is 89 bits); this only stops a malformed
/// manifest from asking for an absurd work cell.
const MAX_INDEX_WORDS: u32 = 16;

/// Written to the resume slot by a cubin whose baked index width disagrees with the
/// `nlimbs` we declared (the ABI-reject sentinel, api.md §1). It is outside the legal
/// resume range `[base, base+count]`, so it can't be confused with a buffer-full
/// watermark.
const ABI_REJECT: u64 = u64::MAX;

impl Abi {
    /// Size of the `start` work cell in u64 words, or `None` for the legacy ABI that
    /// passes the base index by value instead of through a cell.
    fn cell_words(self) -> Option<usize> {
        match self {
            Abi::Legacy { .. } => None,
            Abi::V3 => Some(2),
            Abi::V4 { index_words } => Some(2 + index_words as usize),
        }
    }

    /// Limbs in this job's work-item index. Only v4 can exceed one.
    fn index_words(self) -> u32 {
        match self {
            Abi::V4 { index_words } => index_words,
            Abi::Legacy { .. } | Abi::V3 => 1,
        }
    }
}

/// Build the `start` work cell for one launch of `[base, base+count)` (api.md §1):
/// `[base_0, resume, nlimbs, base_1 …]` under v4, `[base_0, resume]` under v3. Empty
/// under the legacy ABI, which has no cell.
///
/// `resume` is seeded to `base+count` ("ran to the end"); the kernel only ever lowers
/// it. A launch that ends exactly on the limb-0 wrap has `base+count == 2^64`, which
/// no u64 slot can hold, so the seed saturates — see [`read_resume`] for how that one
/// launch is read back.
///
/// Base limbs above 1 are always zero: decryptd tiles a `u128` range, so a job that
/// declares more than two limbs simply runs in the bottom two.
fn work_cell(abi: Abi, base: u128, count: u64) -> Vec<u64> {
    let Some(words) = abi.cell_words() else {
        return Vec::new();
    };
    let mut cell = vec![0u64; words];
    let limb0 = base as u64;
    cell[0] = limb0;
    cell[1] = u64::try_from(u128::from(limb0) + u128::from(count)).unwrap_or(u64::MAX);
    if let Abi::V4 { index_words } = abi {
        cell[2] = u64::from(index_words);
        if index_words >= 2 {
            cell[3] = (base >> 64) as u64;
        }
    }
    cell
}

/// What a finished launch's resume watermark says about how far it got (api.md §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Resume {
    /// Every work-item in `[base, base+count)` was processed and recorded.
    Complete,
    /// The output buffer filled. Everything below this work index is fully recorded;
    /// the rest of the launch's range must be re-run.
    Filled(u128),
    /// The cubin refused the job: it was built for a different index width than the
    /// `nlimbs` we declared. Retrying is pointless.
    Rejected,
}

/// Interpret the watermark a launch left in `start[1]`, given the launch's own
/// `[base, base+count)`.
///
/// The one wrinkle is the launch that ends exactly on the limb-0 wrap: its seed
/// saturated to `u64::MAX`, which is also the rejection sentinel, so that one launch
/// reads back as `Complete` either way. That is the safe reading — a cubin rejects
/// every launch of a job or none, so a rejection has already surfaced on an earlier
/// launch — and the caller re-checks the byte cursor there to catch the overflow the
/// saturated watermark cannot express.
fn read_resume(base: u128, count: u64, resume: u64) -> Resume {
    let limb0 = base as u64;
    let seed = u64::try_from(u128::from(limb0) + u128::from(count)).unwrap_or(u64::MAX);
    if resume == ABI_REJECT && seed != ABI_REJECT {
        return Resume::Rejected;
    }
    if resume >= seed {
        // At or above the seed: untouched (or garbage past the legal range, which
        // clamps the same way) — the launch ran to the end.
        return Resume::Complete;
    }
    // Below `base` can't happen from a legal `atomicMin`; saturating there reports no
    // progress, which the caller handles as "retry this position with fewer items".
    Resume::Filled(base + u128::from(resume.saturating_sub(limb0)))
}

/// Length of the valid prefix of a framed stream (api.md §3 — the record framing is
/// the same under v3 and v4, only the work cell changed): walks
/// `uleb128(len) ‖ payload` records and stops at the terminating zero-length record
/// (the zero-filled tail), at a truncated varint, or at a record that would run past
/// the written region. Trimming each launch's stream to this length lets the
/// per-launch streams be concatenated into one continuous stream for the consumer.
fn framed_len(buf: &[u8]) -> usize {
    walk_framed(buf).0
}

/// Number of records in a framed stream — the count decryptd reports in its logs
/// and submit telemetry (it never looks inside a payload).
pub fn count_framed_records(buf: &[u8]) -> usize {
    walk_framed(buf).1
}

/// Shared walk: returns `(valid byte length, record count)`.
fn walk_framed(buf: &[u8]) -> (usize, usize) {
    let (mut off, mut n) = (0usize, 0usize);
    while off < buf.len() {
        // Decode the uleb128 length prefix.
        let (mut len, mut shift, mut i) = (0u64, 0u32, off);
        loop {
            let Some(&b) = buf.get(i) else {
                return (off, n);
            }; // truncated varint
            i += 1;
            len |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift > 63 {
                return (off, n); // malformed — stop at the last good record
            }
        }
        if len == 0 {
            return (off, n); // zero-length record terminates the stream
        }
        match i.checked_add(len as usize) {
            Some(end) if end <= buf.len() => off = end,
            // A record that overruns the written region: the buffer filled here.
            _ => return (off, n),
        }
        n += 1;
    }
    (off, n)
}

/// Number of CUDA devices visible to the driver (after `CUDA_VISIBLE_DEVICES`).
pub fn device_count() -> Result<i32, String> {
    unsafe {
        check(cuInit(0), "cuInit")?;
        let mut n: i32 = 0;
        check(cuDeviceGetCount(&mut n), "cuDeviceGetCount")?;
        Ok(n)
    }
}

/// Human-readable name of device `ordinal`, queried without creating a context
/// (so the tray can list GPUs cheaply). `cfg`-gated to the GUI build's callers.
#[cfg(all(feature = "gui", any(target_os = "linux", target_os = "windows")))]
pub fn device_name(ordinal: i32) -> Result<String, String> {
    unsafe {
        check(cuInit(0), "cuInit")?;
        let mut dev: CuDevice = 0;
        check(cuDeviceGet(&mut dev, ordinal), "cuDeviceGet")?;
        let mut buf = [0i8; 128];
        check(
            cuDeviceGetName(buf.as_mut_ptr() as *mut c_char, 128, dev),
            "cuDeviceGetName",
        )?;
        Ok(CStr::from_ptr(buf.as_ptr() as *const c_char)
            .to_string_lossy()
            .into_owned())
    }
}

/// PCI bus id of device `ordinal` (e.g. `0000:01:00.0`), the stable key NVML uses
/// to identify the same physical GPU. `None` if it can't be read.
#[cfg(all(feature = "gui", any(target_os = "linux", target_os = "windows")))]
pub fn pci_bus_id(ordinal: i32) -> Option<String> {
    unsafe {
        check(cuInit(0), "cuInit").ok()?;
        let mut dev: CuDevice = 0;
        check(cuDeviceGet(&mut dev, ordinal), "cuDeviceGet").ok()?;
        let mut buf = [0i8; 32];
        if cuDeviceGetPCIBusId(buf.as_mut_ptr() as *mut c_char, buf.len() as i32, dev) != 0 {
            return None;
        }
        Some(
            CStr::from_ptr(buf.as_ptr() as *const c_char)
                .to_string_lossy()
                .into_owned(),
        )
    }
}

/// An initialized CUDA context with a module loaded.
pub struct Gpu {
    ctx: CuContext,
    module: CuModule,
    dev: CuDevice,
    /// Arch tag (`X*10+Y`) of the cubin that actually loaded, e.g. 89 for sm_89.
    arch: u32,
}

impl Gpu {
    /// Init device `ordinal` and load the best cubin for it. Callers pass
    /// `(arch, bytes)` pairs highest-arch-first, where arch is CC `X.Y` encoded as
    /// `X*10+Y`. The created context is current on the *calling thread*, so each
    /// runner thread must call this on its own GPU (see [`crate::run_loop`]).
    ///
    /// Cubins newer than the device are skipped rather than tried: an old driver
    /// (e.g. 550.x / CUDA 12.4) doesn't cleanly reject a cubin for an architecture
    /// it has never heard of — `cuModuleLoadData` faults with SIGILL *inside*
    /// libcuda. So we query the GPU's compute capability first and never hand the
    /// driver anything above it. Same-major-lower cubins that still don't load
    /// (a known arch the driver rejects) fall through to the next candidate.
    pub fn load_first(ordinal: i32, cubins: &[(u32, Vec<u8>)]) -> Result<Gpu, String> {
        unsafe {
            check(cuInit(0), "cuInit")?;
            let mut dev: CuDevice = 0;
            check(cuDeviceGet(&mut dev, ordinal), "cuDeviceGet")?;

            // Device compute capability, encoded to match the `smNN` tags.
            let (mut maj, mut min) = (0i32, 0i32);
            check(
                cuDeviceGetAttribute(&mut maj, CU_DEV_ATTR_CC_MAJOR, dev),
                "cuDeviceGetAttribute(CC_MAJOR)",
            )?;
            check(
                cuDeviceGetAttribute(&mut min, CU_DEV_ATTR_CC_MINOR, dev),
                "cuDeviceGetAttribute(CC_MINOR)",
            )?;
            let gpu_arch = (maj.max(0) as u32) * 10 + (min.max(0) as u32);

            let mut ctx: CuContext = ptr::null_mut();
            check(cuCtxCreate_v2(&mut ctx, 0, dev), "cuCtxCreate")?;
            let mut last = format!("no cubin for sm_{gpu_arch} or older in engine.zip");
            for (arch, cubin) in cubins {
                // Never feed the driver an arch newer than the GPU — it can't run
                // here anyway, and a beyond-driver arch can hard-crash libcuda.
                if *arch > gpu_arch {
                    continue;
                }
                let mut module: CuModule = ptr::null_mut();
                let r = cuModuleLoadData(&mut module, cubin.as_ptr() as *const c_void);
                if r == 0 {
                    return Ok(Gpu {
                        ctx,
                        module,
                        dev,
                        arch: *arch,
                    });
                }
                last = check(r, "cuModuleLoadData").unwrap_err();
            }
            cuCtxDestroy_v2(ctx);
            Err(format!("no cubin loaded on sm_{gpu_arch} ({last})"))
        }
    }

    pub fn device_name(&self) -> String {
        let mut buf = [0i8; 128];
        unsafe {
            if cuDeviceGetName(buf.as_mut_ptr() as *mut c_char, 128, self.dev) == 0 {
                return CStr::from_ptr(buf.as_ptr() as *const c_char)
                    .to_string_lossy()
                    .into_owned();
            }
        }
        "unknown".into()
    }

    pub fn compute_capability(&self) -> (i32, i32) {
        let (mut maj, mut min) = (0i32, 0i32);
        unsafe {
            cuDeviceGetAttribute(&mut maj, CU_DEV_ATTR_CC_MAJOR, self.dev);
            cuDeviceGetAttribute(&mut min, CU_DEV_ATTR_CC_MINOR, self.dev);
        }
        (maj, min)
    }

    /// Arch tag of the cubin that loaded (`X*10+Y`, e.g. 89 for the sm_89 cubin) —
    /// which may be lower than the GPU's own capability if that's the best match.
    pub fn cubin_arch(&self) -> u32 {
        self.arch
    }

    fn function(&self, name: &str) -> Result<CuFunction, String> {
        let cname = CString::new(name).map_err(|e| e.to_string())?;
        let mut f: CuFunction = ptr::null_mut();
        check(
            unsafe { cuModuleGetFunction(&mut f, self.module, cname.as_ptr()) },
            &format!("cuModuleGetFunction({name})"),
        )?;
        Ok(f)
    }
}

impl Drop for Gpu {
    /// Release the module and context. Without this every finished fragment leaks
    /// its CUDA context; after enough fragments `cuCtxCreate` starts failing with
    /// `out of memory` (each context reserves device memory) and no further work
    /// runs. `cuCtxDestroy` alone frees the module too, but unload it explicitly
    /// so the ordering mirrors acquisition.
    fn drop(&mut self) {
        unsafe {
            cuModuleUnload(self.module);
            cuCtxDestroy_v2(self.ctx);
        }
    }
}

/// Run the generic kernel `entry` over `[start, end]` (inclusive), tiling by `tile`
/// items per launch. `data` is the opaque job blob (uploaded once). Returns the raw
/// output records (`out_count * record_size` bytes, concatenated across tiles), and
/// reports per-tile progress via `progress(done, total)`. `gate` is called before
/// each tile launch: it blocks while the worker is paused, so a long fragment stops
/// computing promptly and resumes on the next tile without losing progress.
///
/// `timeout` bounds the *active* compute time of the whole fragment: checked between
/// tiles (paused time excluded), it aborts a fragment that never converges so a
/// runaway job can't pin a GPU forever. The bound is per-tile-granular — a single
/// tile already in flight runs to completion (a blocking `cuCtxSynchronize` can't be
/// interrupted), so keep tiles small enough that one tile is well under the limit.
///
/// `out_cap` bounds only the *on-device* output buffer (one launch's matches), not
/// the job's total result count: the buffer is drained after every launch. It counts
/// records under [`Abi::Legacy`] and bytes under v3/v4. Either way, under-sizing it is
/// safe — no result is ever dropped — but the ABIs recover differently:
///
/// * **v3/v4** use the kernel's resume watermark (api.md §4). A launch whose buffer
///   fills reports the lowest work-index it could not record; everything below that
///   is complete, so the next launch simply restarts there. No work is redone.
/// * **legacy** has no watermark, so an overflowing launch's buffer holds an arbitrary
///   subset (whichever threads won the atomic race) and is unusable. The per-launch
///   item count adapts to the observed match density, targeting ~7/8 fill, and an
///   overflowing launch is recomputed at the corrected size — its work is redone, but
///   the density estimate makes overflow rare (typically once, to calibrate).
///
/// Under either ABI, a single item that alone overflows `out_cap` is a hard error
/// (the cap is too small for the job).
///
/// The range is a `u128` because a v4 work index can exceed 64 bits. Under v3 and
/// legacy — and under a v4 job that declares a single limb — a range reaching past
/// `u64::MAX` is rejected up front rather than silently truncated.
#[allow(clippy::too_many_arguments)]
pub fn run_job(
    gpu: &Gpu,
    entry: &str,
    data: &[u8],
    start: u128,
    end_incl: u128,
    abi: Abi,
    out_cap: u32,
    block: u32,
    tile: u64,
    timeout: Duration,
    mut progress: impl FnMut(u128, u128),
    gate: impl Fn(),
) -> Result<Vec<u8>, String> {
    // Validate the publisher-supplied launch params up front: a bad manifest is a
    // handled error, never a panic (a panic here unwinds the runner thread and
    // takes the whole daemon down). `block == 0` would divide-by-zero below;
    // a zero cap or `record_size` makes the output layout meaningless.
    if block == 0 {
        return Err("manifest block size is 0".into());
    }
    if out_cap == 0 {
        return Err("manifest out_cap is 0".into());
    }
    // Bytes of output buffer per unit of `out_cap`: v3/v4 count bytes directly, legacy
    // counts fixed-size records.
    let cap_unit = match abi {
        Abi::Legacy { record_size } => {
            if record_size == 0 {
                return Err("manifest record_size is 0".into());
            }
            record_size as usize
        }
        Abi::V3 | Abi::V4 { .. } => 1,
    };
    // A v4 job declares its index width; every other ABI is 64-bit by construction.
    let index_words = abi.index_words();
    if index_words == 0 || index_words > MAX_INDEX_WORDS {
        return Err(format!(
            "manifest index_words {index_words} out of range (1..={MAX_INDEX_WORDS})"
        ));
    }
    if index_words < 2 && end_incl > u128::from(u64::MAX) {
        return Err(format!(
            "fragment reaches work index {end_incl}, past the 64-bit index this job \
             declares (format/index_words)"
        ));
    }

    let func = gpu.function(entry)?;
    let d_data = DeviceBuf::from_slice(data)?;
    let d_out = DeviceBuf::alloc(cap_unit * out_cap as usize)?;
    let d_count = DeviceBuf::alloc(4)?;
    // v3/v4 only: the work cell that `start` points at. Its presence is also what
    // marks this run as framed below.
    let d_start = match abi.cell_words() {
        Some(words) => Some(DeviceBuf::alloc(8 * words)?),
        None => None,
    };

    let total = end_incl.saturating_sub(start).saturating_add(1);
    let tile = tile.max(1);
    let mut results = Vec::new();
    let mut done = 0u128;
    let mut cur = start;
    // Wall-clock budget for this fragment, minus any time spent parked in `gate`
    // (a paused worker must not time out). Checked once per tile below.
    let started = Instant::now();
    let mut paused = Duration::ZERO;
    // Items per launch. Under v3/v4 this stays a full tile: a launch that fills the
    // buffer still counts, because the resume watermark says where to pick up, so
    // there is nothing to calibrate. Under legacy it adapts to the observed match
    // density so each launch fills — but does not overflow — the `out_cap` buffer,
    // and it persists across positions: a dense region stays calibrated to a small
    // size instead of resetting to a full tile and re-overflowing (and re-computing)
    // every step. Either way it starts optimistic at a full tile.
    let mut launch = tile;
    while cur <= end_incl {
        let park = Instant::now();
        gate(); // park here while paused (no kernel launched until resumed)
        paused += park.elapsed();
        let active = started.elapsed().saturating_sub(paused);
        if active >= timeout {
            return Err(format!(
                "timed out after {:.0}s (limit {:.0}s) at {done}/{total} items",
                active.as_secs_f64(),
                timeout.as_secs_f64(),
            ));
        }

        // A launch must never cross a limb-0 wrap (api.md §1): that is what keeps every
        // limb above 0 constant across it, so a thread's index is `base + tid` in limb
        // 0 with the high limbs copied, and the watermark stays a plain 64-bit atomic.
        // Splitting there costs nothing — it happens once per 2^64 items.
        let limb0 = cur as u64;
        let to_wrap = (1u128 << 64) - u128::from(limb0);
        let count = ((end_incl - cur).saturating_add(1))
            .min(u128::from(launch))
            .min(to_wrap)
            .max(1) as u64;
        let end_x = cur + u128::from(count);
        d_count.memset0()?;
        if let Some(ds) = &d_start {
            // v3/v4: zero the output buffer — the zero tail *is* the stream terminator,
            // and it also masks the previous launch's bytes past an overflow point —
            // and seed the work cell (resume = "ran to the end").
            d_out.memset0()?;
            let cell: Vec<u8> = work_cell(abi, cur, count)
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            ds.htod(&cell)?;
        }
        // v3/v4 pass a device pointer to the work cell; legacy passes the base index by
        // value. Both are one u64-sized argument slot.
        let mut a_start = match &d_start {
            Some(ds) => ds.ptr,
            None => limb0, // the range was checked to fit a u64 above
        };
        let mut a_count = count;
        let (mut a_data, mut a_dlen) = (d_data.ptr, data.len() as u64);
        let (mut a_out, mut a_oc, mut a_cap) = (d_out.ptr, d_count.ptr, out_cap);
        let mut params: [*mut c_void; 7] = [
            &mut a_start as *mut _ as *mut c_void,
            &mut a_count as *mut _ as *mut c_void,
            &mut a_data as *mut _ as *mut c_void,
            &mut a_dlen as *mut _ as *mut c_void,
            &mut a_out as *mut _ as *mut c_void,
            &mut a_oc as *mut _ as *mut c_void,
            &mut a_cap as *mut _ as *mut c_void,
        ];
        // A too-large tile relative to block can overflow the u32 grid dimension;
        // reject it rather than silently truncating (which would under-compute).
        let grid_u64 = count.div_ceil(block as u64);
        let grid = u32::try_from(grid_u64).map_err(|_| {
            format!("grid {grid_u64} exceeds u32 (tile too large for block {block})")
        })?;
        check(
            unsafe {
                cuLaunchKernel(
                    func,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )?;
        check(unsafe { cuCtxSynchronize() }, "cuCtxSynchronize")?;

        let mut cb = [0u8; 4];
        d_count.dtoh(&mut cb)?;
        let raw = u32::from_le_bytes(cb);

        if let Some(ds) = &d_start {
            // ---- v3/v4: the resume watermark says exactly how far this launch got.
            let mut cell = vec![0u8; ds.len];
            ds.dtoh(&mut cell)?;
            let resume = u64::from_le_bytes(cell[8..16].try_into().unwrap());
            let mut next = match read_resume(cur, count, resume) {
                Resume::Complete => end_x,
                Resume::Filled(n) => n,
                // The cubin was built for a different index width than the manifest
                // declares. Every launch of this job would fail the same way, so stop
                // here rather than burn the whole fragment discovering it repeatedly.
                Resume::Rejected => {
                    return Err(format!(
                        "cubin refused the job (ABI-reject sentinel): it was not built \
                         for the manifest's index_words={index_words}"
                    ));
                }
            };
            // The one launch that ends exactly on the limb-0 wrap could not be seeded
            // with a distinguishable watermark (`limb0+count` is 2^64, which the u64
            // slot can't hold), so an overflow on its very last work-item leaves
            // `resume` looking complete. The byte cursor still shows the buffer filled,
            // and that combination is otherwise impossible — re-run that one item
            // against a fresh buffer rather than risk dropping its records.
            let seed_saturated = u128::from(limb0) + u128::from(count) > u128::from(u64::MAX);
            if next == end_x && raw > out_cap && seed_saturated {
                next = end_x - 1;
            }
            if next == cur {
                // The buffer filled before item `cur` itself could be recorded, so
                // this launch made no progress. Retry the same position with fewer
                // items — a fresh, empty buffer usually clears it. If a single item
                // still can't fit, `out_cap` is genuinely too small for this job.
                if count <= 1 {
                    return Err(format!(
                        "out_cap {out_cap} bytes too small: item {cur} alone \
                         overflows the output buffer"
                    ));
                }
                launch = count / 2;
                continue; // do not advance `cur`/`done`
            }
            // `raw` is the byte cursor, which overshoots `out_cap` when the buffer
            // filled; the framing walk finds the true end of the written prefix.
            //
            // Everything below the watermark is complete, so we keep the whole prefix
            // and restart at `next`. Threads race, so a filled buffer can also hold
            // records for work-indices at or above the watermark, and re-running from
            // it emits those a second time. That's inherent to the ABI (api.md §4
            // prescribes exactly this loop) and decryptd can't dedup — it doesn't know
            // the payload layout, so it can't tell which index a record belongs to.
            // The consumer tolerates a repeated record; a dropped one it could not.
            let written = (raw as usize).min(out_cap as usize);
            if written > 0 {
                let mut buf = vec![0u8; written];
                DeviceBufView {
                    ptr: d_out.ptr,
                    len: written,
                }
                .dtoh(&mut buf)?;
                buf.truncate(framed_len(&buf));
                results.extend_from_slice(&buf);
            }
            launch = tile; // undo any shrink from a no-progress retry
            done += next - cur;
            progress(done.min(total), total);
            cur = next;
            continue;
        }

        // ---- legacy (pre-v3): no watermark, so infer from the record counter.
        // The kernel's counter reflects *every* match, including those past `out_cap`
        // it couldn't write. Treat it as a density sample and re-estimate `launch`
        // for next time to target ~7/8 of `out_cap` — enough headroom that ordinary
        // density variance doesn't tip the following launch into overflow. This is a
        // one-step controller with a stable fixed point at 7/8 fill: `raw == 0` (a
        // sparse region) yields a huge estimate, clamped back up to a full tile.
        let est = (count as u128 * out_cap as u128 * 7 / 8 / raw.max(1) as u128) as u64;
        launch = est.clamp(1, tile);

        if raw > out_cap {
            // Overflow: the buffer holds an arbitrary `out_cap`-sized subset of the
            // matches (whichever threads won the atomic race), so it's unusable.
            // Recompute this same position at the smaller re-estimated size — nothing
            // is dropped, at the cost of redoing this one launch. A single item that
            // still overflows means `out_cap` is genuinely too small for the job.
            if count <= 1 {
                return Err(format!(
                    "out_cap {out_cap} too small: item {cur} alone produced \
                     more than {out_cap} records"
                ));
            }
            // The estimate is already < count whenever raw > out_cap; this guards the
            // integer-rounding corner so a retry always makes progress.
            if launch >= count {
                launch = count / 2;
            }
            continue; // do not advance `cur`/`done`
        }

        if raw > 0 {
            let mut recs = vec![0u8; raw as usize * cap_unit];
            // Read only the populated prefix of the output buffer.
            let mut tmp = DeviceBufView {
                ptr: d_out.ptr,
                len: recs.len(),
            };
            tmp.dtoh(&mut recs)?;
            results.extend_from_slice(&recs);
        }
        done += u128::from(count);
        progress(done.min(total), total);
        cur = end_x;
    }
    Ok(results)
}

// Lightweight view to copy a prefix of an existing device allocation.
struct DeviceBufView {
    ptr: CuDeviceptr,
    len: usize,
}
impl DeviceBufView {
    fn dtoh(&mut self, dst: &mut [u8]) -> Result<(), String> {
        let n = dst.len().min(self.len);
        check(
            unsafe { cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut c_void, self.ptr, n) },
            "cuMemcpyDtoH",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the leaked-context OOM: create and drop a `Gpu` many
    /// times and confirm `cuCtxCreate` keeps succeeding. Before `Drop for Gpu`,
    /// each iteration leaked its context and this loop died with "out of memory"
    /// after a few dozen rounds. Needs a real GPU + a cubin, so it's `#[ignore]`d;
    /// run manually with the cubin path in DECRYPTD_TEST_CUBIN:
    ///   DECRYPTD_TEST_CUBIN=/path/to/x.sm89.cubin cargo test --release -- --ignored gpu_context
    #[test]
    #[ignore]
    fn gpu_context_is_freed_across_runs() {
        let Ok(path) = std::env::var("DECRYPTD_TEST_CUBIN") else {
            panic!("set DECRYPTD_TEST_CUBIN to a cubin matching this GPU's arch");
        };
        let bytes = std::fs::read(&path).expect("read cubin");
        // Tag arch 0 so the "skip cubins newer than the GPU" filter always keeps
        // it; the real cubin must still match this GPU for cuModuleLoadData.
        let cubins = vec![(0u32, bytes)];
        for i in 0..64 {
            let gpu = Gpu::load_first(0, &cubins)
                .unwrap_or_else(|e| panic!("iteration {i}: load_first failed: {e}"));
            // Touch it so the context is really used, then drop at end of scope.
            let _ = gpu.compute_capability();
            drop(gpu);
        }
    }

    /// Frame a payload the way a kernel's record-emit primitive does (v3 and v4 alike).
    fn frame(payload: &[u8]) -> Vec<u8> {
        let (mut v, mut len) = (Vec::new(), payload.len() as u64);
        while len >= 0x80 {
            v.push((len as u8 & 0x7f) | 0x80);
            len >>= 7;
        }
        v.push(len as u8);
        v.extend_from_slice(payload);
        v
    }

    /// The framed stream walk (api.md §3) must find the exact end of the written region
    /// so per-launch streams concatenate cleanly: it stops at the zero-length record
    /// that the zero-filled tail forms, and never reads into the untouched remainder.
    #[test]
    fn framed_walk_stops_at_zero_terminator() {
        let mut buf = Vec::new();
        for i in 0u64..3 {
            buf.extend_from_slice(&frame(&i.to_le_bytes())); // cracker payload: u64 index
        }
        let used = buf.len();
        assert_eq!(used, 3 * 9, "uleb128(8) + 8 bytes per record");
        buf.resize(used + 512, 0); // the zero-filled tail of the output buffer
        assert_eq!(framed_len(&buf), used);
        assert_eq!(count_framed_records(&buf), 3);
        // Trimming to `framed_len` is what makes concatenation valid.
        let mut joined = buf[..framed_len(&buf)].to_vec();
        joined.extend_from_slice(&buf[..framed_len(&buf)]);
        assert_eq!(count_framed_records(&joined), 6);
    }

    /// A multi-byte length prefix (payload >= 128 bytes) must decode as standard
    /// unsigned LEB128, not be mistaken for two records.
    #[test]
    fn framed_walk_handles_multibyte_lengths() {
        let big = vec![0xabu8; 300];
        let buf = frame(&big);
        assert_eq!(buf[0..2], [0xac, 0x02], "uleb128(300)");
        assert_eq!(framed_len(&buf), buf.len());
        assert_eq!(count_framed_records(&buf), 1);
    }

    /// When a launch overflows, the byte cursor overshoots and the last record is cut
    /// off mid-payload (or mid-varint). The walk must drop that partial record rather
    /// than emit a corrupt one — the resume watermark re-runs it next launch.
    #[test]
    fn framed_walk_drops_a_truncated_tail_record() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&frame(&7u64.to_le_bytes()));
        let complete = buf.len();
        buf.extend_from_slice(&frame(&9u64.to_le_bytes()));
        for cut in 1..=8 {
            let partial = &buf[..buf.len() - cut]; // payload cut short
            assert_eq!(framed_len(partial), complete, "cut {cut}");
            assert_eq!(count_framed_records(partial), 1, "cut {cut}");
        }
        // Cut so only the second record's length prefix survives, then nothing at all.
        assert_eq!(framed_len(&buf[..complete + 1]), complete);
        assert_eq!(framed_len(&buf[..complete]), complete);
    }

    /// An unterminated varint (high bit set forever) must not loop or over-read.
    #[test]
    fn framed_walk_rejects_a_malformed_varint() {
        let buf = vec![0xffu8; 32];
        assert_eq!(framed_len(&buf), 0);
        assert_eq!(count_framed_records(&buf), 0);
    }

    /// The v4 work cell (api.md §1): `[base_0, resume, nlimbs, base_1 …]`, with the
    /// base index split into little-endian 64-bit limbs and `resume` seeded to
    /// `base+count`. v3's cell is the same minus the self-describing tail.
    #[test]
    fn work_cell_layout() {
        assert_eq!(work_cell(Abi::V3, 100, 8), vec![100, 108]);
        assert_eq!(
            work_cell(Abi::V4 { index_words: 1 }, 100, 8),
            vec![100, 108, 1]
        );
        // A Coldcard Mk4-style 89-bit index: limb 1 carries the RTC state.
        let base = (7u128 << 64) | 100;
        assert_eq!(
            work_cell(Abi::V4 { index_words: 2 }, base, 8),
            vec![100, 108, 2, 7],
        );
        // More limbs than a u128 range can fill: the rest are zero, not garbage.
        assert_eq!(
            work_cell(Abi::V4 { index_words: 4 }, base, 8),
            vec![100, 108, 4, 7, 0, 0],
        );
        // The legacy ABI has no cell at all — `start` is passed by value.
        assert!(work_cell(Abi::Legacy { record_size: 8 }, 100, 8).is_empty());
    }

    /// The watermark a launch leaves behind decodes to one of three outcomes, and the
    /// "filled" one must translate back into a full-width `u128` work index.
    #[test]
    fn resume_watermark_outcomes() {
        // Untouched seed: the launch ran to the end.
        assert_eq!(read_resume(100, 8, 108), Resume::Complete);
        // Buffer filled at 104: everything below it is recorded.
        assert_eq!(read_resume(100, 8, 104), Resume::Filled(104));
        // Filled at the very first item — no progress; the caller retries smaller.
        assert_eq!(read_resume(100, 8, 100), Resume::Filled(100));
        // The watermark is in limb-0 units, so the high limbs come back from `base`.
        let base = (7u128 << 64) | 100;
        assert_eq!(
            read_resume(base, 8, 104),
            Resume::Filled((7u128 << 64) | 104)
        );
        // Width mismatch: the cubin refused the job outright.
        assert_eq!(read_resume(100, 8, u64::MAX), Resume::Rejected);
        // Garbage below the base can't come from a legal atomicMin; report no progress
        // rather than rewinding the range.
        assert_eq!(read_resume(100, 8, 3), Resume::Filled(100));
    }

    /// A launch ending exactly on the limb-0 wrap can't hold `base+count == 2^64` in
    /// the u64 resume slot, so the seed saturates. It must still read back as complete
    /// — and must not be mistaken for a rejection, since `u64::MAX` is the seed here.
    #[test]
    fn resume_watermark_at_the_limb_wrap() {
        let base = u128::from(u64::MAX) - 9; // [2^64-10, 2^64)
        assert_eq!(work_cell(Abi::V4 { index_words: 2 }, base, 10)[1], u64::MAX);
        assert_eq!(read_resume(base, 10, u64::MAX), Resume::Complete);
        assert_eq!(
            read_resume(base, 10, u64::MAX - 5),
            Resume::Filled(base + 4)
        );
        // Only the watermark wraps, not the index: a launch sitting above 2^64 but
        // inside its limb seeds an ordinary in-range value and reads back normally.
        let high = (1u128 << 64) + 100;
        assert_eq!(work_cell(Abi::V4 { index_words: 2 }, high, 10)[1], 110);
        assert_eq!(read_resume(high, 10, 110), Resume::Complete);
        assert_eq!(read_resume(high, 10, u64::MAX), Resume::Rejected);
    }

    /// Verifies the density-adaptive overflow path: with an `out_cap` far smaller
    /// than a launch's match count, `run_job` must still return *every* record by
    /// recomputing at the corrected size — none silently dropped, none duplicated.
    ///
    /// Needs a real GPU and a companion `emit` cubin implementing the kernel ABI so
    /// that it emits K records per item — K read from the first 4 bytes of the blob,
    /// each record being [(u32) item, (u32) ordinal]. Run manually:
    ///   DECRYPTD_TEST_CUBIN=/path/to/emit.sm120.cubin \
    ///     cargo test --release -- --ignored out_cap_overflow
    #[test]
    #[ignore]
    fn out_cap_overflow_recovers_all_records() {
        let Ok(path) = std::env::var("DECRYPTD_TEST_CUBIN") else {
            panic!("set DECRYPTD_TEST_CUBIN to the emit cubin matching this GPU");
        };
        let bytes = std::fs::read(&path).expect("read cubin");
        let cubins = vec![(0u32, bytes)];
        let gpu = Gpu::load_first(0, &cubins).expect("load_first");

        const N: u64 = 10_000; // items
        const K: u32 = 7; // records emitted per item
        let data = K.to_le_bytes().to_vec();
        let expected = N as usize * K as usize;

        let out = run_job(
            &gpu,
            "emit",
            &data,
            0,
            u128::from(N - 1),
            Abi::Legacy { record_size: 8 }, // [u32 item, u32 ordinal]
            1000,    // out_cap: deliberately << N*K, forcing many subdivisions
            128,     // block
            1 << 20, // tile: whole range fits one tile, so the first launch overflows
            Duration::from_secs(60),
            |_, _| {},
            || {},
        )
        .expect("run_job");

        assert_eq!(out.len(), expected * 8, "byte length mismatch");
        let mut seen = std::collections::HashSet::new();
        for rec in out.chunks_exact(8) {
            let item = u32::from_le_bytes(rec[0..4].try_into().unwrap());
            let ord = u32::from_le_bytes(rec[4..8].try_into().unwrap());
            assert!((item as u64) < N, "item {item} out of range");
            assert!(ord < K, "ordinal {ord} out of range");
            assert!(seen.insert((item, ord)), "duplicate record ({item},{ord})");
        }
        assert_eq!(seen.len(), expected, "missing records after subdivision");
    }
}
