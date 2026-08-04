//! Minimal CUDA Driver-API wrapper for the generic launch path. decryptd knows
//! nothing about the kernel's job — it uploads an opaque data blob, launches a
//! kernel with the fixed ABI below over a range, and reads back the output records.
//!
//! Kernel ABI v5 (the contract every current decryptd cubin implements; the full
//! spec is the publisher's `api.md`):
//! ```c
//! extern "C" __global__ void <entry>(
//!     unsigned long long* start,   // in/out: the work cell, 2 + nfields x u64
//!     unsigned long long count,    // box cells (odometer steps) in this launch
//!     const unsigned char* data,   // the job data blob (device) — carries the box
//!     unsigned long long data_len,
//!     unsigned char* out,          // output byte-stream buffer (device)
//!     unsigned int* out_count,     // atomic BYTE cursor
//!     unsigned int out_cap);       // capacity in BYTES
//! ```
//! The work space is a **box** of independent per-axis ranges, walked as a mixed-radix
//! odometer. Every earlier ABI addressed work by one flat integer index, which real
//! search spaces don't fit: their axes are not powers of two, so flattening one pads
//! each axis up to the next power of two and burns the difference on cells that don't
//! exist — and it overflows 64 bits once enough axes stack. The cell carries a per-axis
//! cursor instead of an index, and the resume watermark counts *steps of this launch*
//! rather than naming an absolute position.
//!
//! The platform hands the box to decryptd directly, as `Job.Bounds` alongside each
//! fragment's `Start`/`Steps`, so a fragment is a contiguous run in the kernel's own
//! walk order and tiling it is just advancing the cursor. The job blob carries the same
//! axes for the kernel's carry arithmetic — decryptd never opens it. Axes that exist
//! only to amortise an expensive stage across their span are not in the box at all:
//! they are baked into the cubin as inner loops and reported in the record's trailing
//! fields, so the box's axis count is the whole story (api.md §1.3).
//!
//! `start[1]` echoes that count, and a cubin whose blob disagrees — or whose box is
//! wider than it can hold — rejects the job instead of computing garbage.
//!
//! **v5 only.** The pre-v5 ABIs are gone: v3 (`[base, resume]`, 64-bit index) and pre-v3
//! (`start` by value, `out_count`/`out_cap` in fixed-size records) are no longer run,
//! and v4 was superseded before any job was dispatched. A manifest whose `format` is not
//! 5, or a fragment carved as a flat range, is refused rather than misread — every one
//! of those seeded a differently shaped work cell.

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

/// Upper bound on a box's axis count. The deepest spaces publishers describe today are
/// well under this; it only stops a malformed spec from making us allocate an absurd
/// work cell.
const MAX_BOX_FIELDS: usize = 64;

/// Written to the resume slot by a cubin that cannot run the job — the `nfields` we
/// declared is not the one its blob carries, or the box has more axes than it can hold
/// (api.md §4). Unambiguous because a launch's `count` is always below it, so it can
/// never be confused with a buffer-full watermark.
const ABI_REJECT: u64 = u64::MAX;

/// The axes of a job's work space: one inclusive `[lo, hi]` range each, held in
/// **kernel order** — axis 0 first, cycling fastest (api.md §2).
///
/// This comes from the platform's `Job.Bounds`, so decryptd never opens the job blob.
/// The blob carries the same axes for the kernel's own carry arithmetic; the axes that
/// amortise an expensive stage are baked into the cubin as inner loops and are not part
/// of the box at all, so its axis count is the whole story (api.md §1.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkBox {
    axes: Vec<(u64, u64)>,
}

impl WorkBox {
    /// Parse a `Job.Bounds` spec: `"lo-hi/lo-hi/…"`, one field per thread axis.
    ///
    /// The platform writes it **leftmost-most-significant** — the rightmost field is
    /// the one that steps every cell and carries left — which is the reverse of the
    /// kernel's axis order, so the fields are stored reversed.
    pub fn parse_bounds(spec: &str) -> Result<WorkBox, String> {
        let fields: Vec<&str> = spec.split('/').collect();
        if fields.is_empty() || fields.len() > MAX_BOX_FIELDS {
            return Err(format!(
                "bounds {spec:?} has {} axes (expected 1..={MAX_BOX_FIELDS})",
                fields.len()
            ));
        }
        let mut axes = Vec::with_capacity(fields.len());
        for (i, field) in fields.iter().enumerate() {
            let f = field.trim();
            let (lo, hi) = f
                .split_once('-')
                .ok_or_else(|| format!("bounds axis {i} {f:?} is not `lo-hi`"))?;
            let parse = |s: &str, what: &str| -> Result<u64, String> {
                s.trim()
                    .parse::<u64>()
                    .map_err(|e| format!("bounds axis {i} {what} {s:?}: {e}"))
            };
            let (lo, hi) = (parse(lo, "low")?, parse(hi, "high")?);
            // api.md §1.5: `lo <= hi` keeps the axis non-empty, and `hi < u64::MAX`
            // keeps its radix from overflowing to 0. A spec breaking either would make
            // the odometer silently wrong rather than loudly broken.
            if lo > hi {
                return Err(format!("bounds axis {i} is empty: {lo} > {hi}"));
            }
            if hi == u64::MAX {
                return Err(format!(
                    "bounds axis {i} ends at u64::MAX; it must be split"
                ));
            }
            axes.push((lo, hi));
        }
        axes.reverse(); // most-significant-first on the wire -> axis 0 first here
        Ok(WorkBox { axes })
    }

    /// Parse a `Fragment.Start` position: one absolute value per axis, in the same
    /// leftmost-most-significant order as the bounds, and inside them.
    pub fn parse_position(&self, spec: &str) -> Result<Vec<u64>, String> {
        let mut pos = Vec::with_capacity(self.axes.len());
        for (i, field) in spec.split('/').enumerate() {
            let f = field.trim();
            pos.push(
                f.parse::<u64>()
                    .map_err(|e| format!("start axis {i} {f:?} is not a bare value: {e}"))?,
            );
        }
        if pos.len() != self.axes.len() {
            return Err(format!(
                "start {spec:?} has {} axes but the bounds have {}",
                pos.len(),
                self.axes.len()
            ));
        }
        pos.reverse(); // same wire order as the bounds
        for (i, (&p, &(lo, hi))) in pos.iter().zip(&self.axes).enumerate() {
            if p < lo || p > hi {
                return Err(format!(
                    "start axis {i} value {p} is outside its bound {lo}-{hi}"
                ));
            }
        }
        Ok(pos)
    }

    /// Axes in the box — the `nfields` the work cell declares, and the length of a
    /// cursor.
    pub fn nfields(&self) -> usize {
        self.axes.len()
    }

    /// Span of axis `i` — its radix in the odometer.
    fn radix(&self, i: usize) -> u128 {
        let (lo, hi) = self.axes[i];
        u128::from(hi - lo) + 1
    }

    /// Total cells in the box — the unit `Fragment.Steps` and the manifest's `tile`
    /// count in.
    pub fn cells(&self) -> Result<u128, String> {
        (0..self.axes.len()).try_fold(1u128, |acc, i| {
            acc.checked_mul(self.radix(i))
                .ok_or_else(|| "box has more than 2^128 cells".to_string())
        })
    }

    /// Largest radix in the box — the term api.md §1.5 bounds a launch's `count`
    /// against, so the kernel's per-thread `(r_i - 1) + carry` cannot overflow.
    fn max_radix(&self) -> u128 {
        (0..self.axes.len())
            .map(|i| self.radix(i))
            .max()
            .unwrap_or(1)
    }

    /// `pos` advanced `delta` cells through the box — the host-side twin of the
    /// kernel's per-thread decode (api.md §1.4), computed digit-wise so a box past
    /// 2^64 stays addressable without ever forming a flat offset.
    ///
    /// `None` means the walk carried off the end of the box, which is what ends a run.
    fn advance(&self, pos: &[u64], delta: u128) -> Option<Vec<u64>> {
        let mut out = pos.to_vec();
        let mut carry = delta;
        for (i, v) in out.iter_mut().enumerate() {
            if carry == 0 {
                break; // axes never reached keep their current value
            }
            let (lo, _) = self.axes[i];
            let r = self.radix(i);
            let n = u128::from(*v - lo) + carry;
            *v = lo + (n % r) as u64;
            carry = n / r;
        }
        (carry == 0).then_some(out)
    }
}

/// The work a claimed fragment covers: `steps` cells of `bounds`' odometer starting at
/// `pos` (`Decrypt/Job:pullOne` hands these out as `Start`/`Steps` beside `Job.Bounds`).
///
/// A fragment is a *contiguous run* in the kernel's own walk order, so tiling it is
/// just advancing the cursor.
#[derive(Clone, Debug)]
pub struct Work {
    pub bounds: WorkBox,
    pub pos: Vec<u64>,
    pub steps: u64,
}

impl std::fmt::Display for Work {
    /// How a fragment is named in logs: its start position and length, most-significant
    /// axis first — the platform's own spelling, so a log line can be matched against a
    /// `Decrypt/Job/Fragment` row.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let wire: Vec<String> = self.pos.iter().rev().map(|v| v.to_string()).collect();
        write!(f, "{} +{}", wire.join("/"), self.steps)
    }
}

/// Build the `start` work cell for a launch of `count` cells from `pos` (api.md §1.2):
/// `[resume = count, nfields, pos…]`.
///
/// The resume slot is *seeded* by the publisher with the launch's step count and only
/// ever lowered by the kernel. Being launch-relative is what removed v4's one
/// unresolvable corner: there is no absolute end value that might not fit the slot, and
/// "ran to the end" is just `count`, not a distinguished sentinel.
fn work_cell(pos: &[u64], count: u64) -> Vec<u64> {
    let mut cell = Vec::with_capacity(2 + pos.len());
    cell.push(count);
    cell.push(pos.len() as u64);
    cell.extend_from_slice(pos);
    cell
}

/// How many steps of a launch of `count` cells completed, read back from the resume
/// slot (api.md §4). `Err` means the cubin refused the job — the `nfields` we declared
/// is not the one its blob carries, or the box is wider than it can hold — so every
/// launch would fail identically.
fn steps_done(count: u64, resume: u64) -> Result<u64, ()> {
    if resume == ABI_REJECT {
        return Err(());
    }
    // Clamp: a kernel that leaves the slot untouched (or writes nonsense above the
    // seed) must not carry the cursor past what the launch actually covered.
    Ok(resume.min(count))
}

/// Length of the valid prefix of a framed stream (api.md §3): walks
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

/// Run the generic kernel `entry` over the fragment `work` describes, tiling by `tile`
/// items per launch. `data` is the opaque job blob (uploaded once). Returns the raw
/// output records (concatenated across tiles), and reports per-tile progress via
/// `progress(done, total)`. `gate` is called before each tile launch: it blocks while
/// the worker is paused, so a long fragment stops computing promptly and resumes on
/// the next tile without losing progress.
///
/// `timeout` bounds the *active* compute time of the whole fragment: checked between
/// tiles (paused time excluded), it aborts a fragment that never converges so a
/// runaway job can't pin a GPU forever. The bound is per-tile-granular — a single
/// tile already in flight runs to completion (a blocking `cuCtxSynchronize` can't be
/// interrupted), so keep tiles small enough that one tile is well under the limit.
///
/// `out_cap` bounds only the *on-device* output buffer in bytes (what one launch emits),
/// not the job's total result count: the buffer is drained after every launch.
/// Under-sizing it is safe — no result is ever dropped — because the kernel's resume
/// watermark (api.md §4) reports how far a launch whose buffer filled actually got.
/// Everything before that is complete, so the next launch restarts there and no work is
/// redone. A single cell that alone overflows `out_cap` is a hard error: the cap is too
/// small for the job.
#[allow(clippy::too_many_arguments)]
pub fn run_job(
    gpu: &Gpu,
    entry: &str,
    data: &[u8],
    work: &Work,
    out_cap: u32,
    block: u32,
    tile: u64,
    timeout: Duration,
    mut progress: impl FnMut(u128, u128),
    gate: impl Fn(),
) -> Result<Vec<u8>, String> {
    // Validate the publisher-supplied launch params up front: a bad manifest is a
    // handled error, never a panic (a panic here unwinds the runner thread and takes
    // the whole daemon down). `block == 0` would divide-by-zero below; a zero cap makes
    // the output buffer meaningless.
    if block == 0 {
        return Err("manifest block size is 0".into());
    }
    if out_cap == 0 {
        return Err("manifest out_cap is 0".into());
    }

    let func = gpu.function(entry)?;
    let d_data = DeviceBuf::from_slice(data)?;
    let d_out = DeviceBuf::alloc(out_cap as usize)?;
    let d_count = DeviceBuf::alloc(4)?;
    // The work cell `start` points at: `[resume, nfields, pos…]`, one cursor word per
    // box axis.
    let d_start = DeviceBuf::alloc(8 * (2 + work.bounds.nfields()))?;

    // Cells this fragment covers. A run can never be longer than the whole box, so an
    // over-issued `Steps` is clamped rather than trusted — the odometer's carry stops it
    // either way (api.md §1.4), but the progress denominator should be honest.
    let total = u128::from(work.steps).min(work.bounds.cells()?);
    // api.md §1.5 bounds a launch: `count < u64::MAX` keeps the rejection sentinel
    // unambiguous, and `count <= u64::MAX - max(r_i)` keeps the kernel's per-thread
    // `(r_i - 1) + carry` from overflowing. No real tile comes near either, but a
    // manifest is publisher input — clamp it rather than trust it.
    let tile_cap =
        u64::try_from(u128::from(u64::MAX) - work.bounds.max_radix()).unwrap_or(u64::MAX - 1);
    let tile = tile.clamp(1, tile_cap.max(1));
    let mut results = Vec::new();
    let mut done = 0u128;
    // Wall-clock budget for this fragment, minus any time spent parked in `gate`
    // (a paused worker must not time out). Checked once per tile below.
    let started = Instant::now();
    let mut paused = Duration::ZERO;
    // Cells per launch. Normally a full tile: a launch that fills the output buffer
    // still counts, because the resume watermark says where to pick up, so there is
    // nothing to calibrate. It only shrinks to break a launch that made no progress at
    // all (see below), and is restored as soon as one succeeds.
    let mut launch = tile;
    while done < total {
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

        // Where this launch starts: the odometer cursor `done` cells into the fragment.
        let Some(cursor) = work.bounds.advance(&work.pos, done) else {
            // The odometer carried off the end of the box. The fragment claimed more
            // cells than the box holds; everything that exists has been run.
            break;
        };
        let count = (total - done).min(u128::from(launch)).max(1) as u64;
        d_count.memset0()?;
        // Zero the output buffer — the zero tail *is* the stream terminator, and it also
        // masks the previous launch's bytes past an overflow point — then seed the work
        // cell (the kernel only ever lowers the resume slot).
        d_out.memset0()?;
        let cell: Vec<u8> = work_cell(&cursor, count)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        d_start.htod(&cell)?;

        let mut a_start = d_start.ptr;
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

        // The resume watermark says exactly how far this launch got.
        let mut cell = vec![0u8; d_start.len];
        d_start.dtoh(&mut cell)?;
        let resume = u64::from_le_bytes(cell[..8].try_into().unwrap());
        // A rejection means this cubin cannot run this job at all — the `nfields` we
        // declared is not the one its blob carries, or the box is wider than it holds.
        // Every launch would fail the same way, so stop rather than burn the whole
        // fragment discovering it repeatedly.
        let steps = steps_done(count, resume).map_err(|()| {
            format!(
                "cubin refused the job (ABI-reject sentinel): it was not built for a \
                 {}-axis work space",
                work.bounds.nfields()
            )
        })?;
        if steps == 0 {
            // The buffer filled before the launch's very first cell could be recorded,
            // so it made no progress. Retry the same position with fewer cells — a
            // fresh, empty buffer usually clears it. If a single cell still can't fit,
            // `out_cap` is genuinely too small for this job.
            if count <= 1 {
                return Err(format!(
                    "out_cap {out_cap} bytes too small: cell {done} of this fragment \
                     alone overflows the output buffer"
                ));
            }
            launch = count / 2;
            continue; // do not advance `done`
        }
        // `raw` is the byte cursor, which overshoots `out_cap` when the buffer filled;
        // the framing walk finds the true end of the written prefix.
        //
        // Everything before the watermark is complete, so we keep the whole prefix and
        // restart there. Threads race, so a filled buffer can also hold records for
        // steps at or past the watermark, and re-running from it emits those a second
        // time. That's inherent to the ABI (api.md §4 prescribes exactly this loop) and
        // decryptd can't dedup — it doesn't know the payload layout, so it can't tell
        // which cell a record belongs to. The consumer tolerates a repeated record; a
        // dropped one it could not.
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
        done += u128::from(steps);
        progress(done.min(total), total);
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

    /// Frame a payload the way a kernel's record-emit primitive does.
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
            buf.extend_from_slice(&frame(&i.to_le_bytes())); // an 8-byte payload
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

    /// `Job.Bounds` is written **leftmost most significant** — the rightmost field is
    /// the one that steps every cell — which is the reverse of the kernel's axis order.
    /// Getting this backwards would walk the space transposed, so pin it.
    #[test]
    fn work_box_reverses_the_wire_axis_order() {
        // A `[pad, ssr, tod]` space as the platform spells it: tod steps fastest.
        let b = WorkBox::parse_bounds("0-4999999/0-254/0-86399").unwrap();
        assert_eq!(b.nfields(), 3);
        // Axis 0 is the rightmost field.
        assert_eq!(b.axes, vec![(0, 86_399), (0, 254), (0, 4_999_999)]);
        assert_eq!(b.cells().unwrap(), 86_400 * 255 * 5_000_000);

        // A single-axis space is the degenerate case.
        let b = WorkBox::parse_bounds("0-999999").unwrap();
        assert_eq!((b.nfields(), b.cells().unwrap()), (1, 1_000_000));
    }

    /// A start position is one bare value per axis, in the same wire order as the
    /// bounds, and inside them.
    #[test]
    fn work_box_parses_a_start_position() {
        let b = WorkBox::parse_bounds("4-8/1-255/0-65536").unwrap();
        // Reversed to kernel order: axis 0 is the `0-65536` field.
        assert_eq!(b.parse_position("4/1/0").unwrap(), vec![0, 1, 4]);
        assert_eq!(b.parse_position("7/200/4096").unwrap(), vec![4096, 200, 7]);

        assert!(b.parse_position("4/1").is_err(), "too few axes");
        assert!(b.parse_position("4/1/0/9").is_err(), "too many axes");
        assert!(b.parse_position("9/1/0").is_err(), "axis 2 above its bound");
        assert!(b.parse_position("4/0/0").is_err(), "axis 1 below its bound");
        assert!(b.parse_position("4-8/1/0").is_err(), "a range, not a value");
    }

    /// The odometer: axis 0 steps every cell and carries left, and each digit is an
    /// absolute axis value — not an offset from the bound (api.md §1.4).
    #[test]
    fn work_box_advances_least_significant_first() {
        // Radices 7 and 10 (`5-11` steps fastest), both off zero so absolute values show.
        let b = WorkBox::parse_bounds("100-109/5-11").unwrap();
        assert_eq!(b.cells().unwrap(), 70);
        let origin = b.parse_position("100/5").unwrap();
        assert_eq!(origin, vec![5, 100]);
        assert_eq!(b.advance(&origin, 0), Some(vec![5, 100]));
        assert_eq!(b.advance(&origin, 1), Some(vec![6, 100])); // axis 0 steps
        assert_eq!(b.advance(&origin, 6), Some(vec![11, 100])); // last of axis 0
        assert_eq!(b.advance(&origin, 7), Some(vec![5, 101])); // carry into axis 1
        assert_eq!(b.advance(&origin, 69), Some(vec![11, 109])); // last cell
        // Carrying off the end of the box is what ends a run.
        assert_eq!(b.advance(&origin, 70), None);

        // Advancing from mid-box, not just the origin.
        let mid = b.parse_position("103/9").unwrap();
        assert_eq!(b.advance(&mid, 3), Some(vec![5, 104]));

        // A box far past 2^64 stays addressable — no flat offset is ever formed. Ten
        // axes of radix 95 is 95^10 = 2^66.4 cells, well beyond any single integer.
        let wide = WorkBox::parse_bounds(&["0-94"; 10].join("/")).unwrap();
        assert_eq!(wide.cells().unwrap(), 95u128.pow(10));
        assert!(wide.cells().unwrap() > u128::from(u64::MAX));
        let zero = vec![0u64; 10];
        assert_eq!(
            wide.advance(&zero, 95 * 95),
            Some(vec![0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        );
        assert_eq!(wide.advance(&zero, 95u128.pow(10) - 1), Some(vec![94; 10]));
        assert_eq!(wide.advance(&zero, 95u128.pow(10)), None);
    }

    /// A bounds spec that is malformed, or breaks api.md §1.5, must be a handled error
    /// — the odometer would be silently wrong rather than loudly broken.
    #[test]
    fn work_box_rejects_malformed_bounds() {
        assert!(WorkBox::parse_bounds("").is_err(), "empty spec");
        assert!(WorkBox::parse_bounds("0-9/").is_err(), "empty axis field");
        assert!(WorkBox::parse_bounds("0..9").is_err(), "not `lo-hi`");
        assert!(WorkBox::parse_bounds("0-9/x-3").is_err(), "not a number");
        assert!(WorkBox::parse_bounds("-5-9").is_err(), "negative low");
        // lo > hi is an empty axis; hi == u64::MAX overflows the radix to 0.
        assert!(WorkBox::parse_bounds("9-0").is_err(), "empty axis");
        assert!(
            WorkBox::parse_bounds(&format!("0-{}", u64::MAX)).is_err(),
            "full u64 axis"
        );
    }

    /// A fragment reads back in the platform's own spelling, so a log line can be
    /// matched against a `Decrypt/Job/Fragment` row.
    #[test]
    fn work_labels_match_the_wire_spelling() {
        let bounds = WorkBox::parse_bounds("4-8/1-255/0-65536").unwrap();
        let pos = bounds.parse_position("7/200/4096").unwrap();
        let work = Work {
            bounds,
            pos,
            steps: 4096,
        };
        assert_eq!(work.to_string(), "7/200/4096 +4096");
    }

    /// The work cell (api.md §1.2) is `[resume, nfields, pos…]`, with `resume` seeded to
    /// the launch's step count and the cursor in kernel axis order.
    #[test]
    fn work_cell_layout() {
        let b = WorkBox::parse_bounds("100-109/5-11").unwrap();
        let origin = b.parse_position("100/5").unwrap();
        // 12 cells in: axis 0 (radix 7) wrapped once, so axis 1 is at 101.
        let cursor = b.advance(&origin, 12).unwrap();
        assert_eq!(cursor, vec![10, 101]);
        assert_eq!(work_cell(&cursor, 8), vec![8, 2, 10, 101]);
    }

    /// The watermark says how many steps of the launch completed — it is
    /// launch-relative, so "ran to the end" is just `count` rather than a sentinel.
    #[test]
    fn resume_watermark_reports_steps_done() {
        // Untouched seed: the launch ran to the end.
        assert_eq!(steps_done(8, 8), Ok(8));
        // Buffer filled after 5 steps; everything before that is recorded.
        assert_eq!(steps_done(8, 5), Ok(5));
        // Filled before the first step — no progress; the caller retries smaller.
        assert_eq!(steps_done(8, 0), Ok(0));
        // Garbage above the seed can't come from a legal atomicMin; clamp to the count
        // rather than carrying the cursor past the launch.
        assert_eq!(steps_done(8, 99), Ok(8));
        // The cubin cannot run this job's box shape: refused outright.
        assert_eq!(steps_done(8, ABI_REJECT), Err(()));
    }

    /// Verifies the overflow/resume path end to end: with an `out_cap` far smaller than
    /// a launch's record count, `run_job` must still return **every** record, restarting
    /// each time at the watermark the kernel left behind.
    ///
    /// Note what is and isn't guaranteed. No record is ever dropped — that's the point
    /// of the watermark. Records *can* repeat: threads race, so a filled buffer may
    /// already hold records at or past the watermark, and re-running from it emits them
    /// again (api.md §4). So this asserts full coverage, not exactly-once.
    ///
    /// Needs a real GPU and a companion `emit` cubin implementing the v5 ABI so that it
    /// emits K records per cell — K read from the first 4 bytes of the blob, each record
    /// framed as `uleb128(8) ‖ [(u32) cell, (u32) ordinal]`. Run manually:
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

        const N: u64 = 10_000; // cells
        const K: u32 = 7; // records emitted per cell
        let data = K.to_le_bytes().to_vec();
        let bounds = WorkBox::parse_bounds(&format!("0-{}", N - 1)).unwrap();
        let pos = bounds.parse_position("0").unwrap();

        let out = run_job(
            &gpu,
            "emit",
            &data,
            &Work {
                bounds,
                pos,
                steps: N,
            },
            1000,    // out_cap: deliberately << N*K*9 bytes, forcing many resumes
            128,     // block
            1 << 20, // tile: the whole run fits one tile, so the first launch overflows
            Duration::from_secs(60),
            |_, _| {},
            || {},
        )
        .expect("run_job");

        // Walk the framed stream and collect every (cell, ordinal) pair it carries.
        let mut seen = std::collections::HashSet::new();
        let (mut off, mut n) = (0usize, 0usize);
        while off < out.len() {
            let len = out[off] as usize; // payloads are 8 bytes, so one length byte
            if len == 0 {
                break;
            }
            assert_eq!(len, 8, "record {n} has an unexpected payload length");
            let rec = &out[off + 1..off + 1 + len];
            let cell = u32::from_le_bytes(rec[0..4].try_into().unwrap());
            let ord = u32::from_le_bytes(rec[4..8].try_into().unwrap());
            assert!((cell as u64) < N, "cell {cell} out of range");
            assert!(ord < K, "ordinal {ord} out of range");
            seen.insert((cell, ord));
            off += 1 + len;
            n += 1;
        }
        assert_eq!(
            seen.len(),
            N as usize * K as usize,
            "records missing after {n} emitted across resumes",
        );
    }
}
