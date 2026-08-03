//! Minimal CUDA Driver-API wrapper for the generic launch path. decryptd knows
//! nothing about the kernel's job — it uploads an opaque data blob, launches a
//! kernel with the fixed ABI below over a range, and reads back the output records.
//!
//! Kernel ABI v5 (the contract every current decryptd cubin implements; the full
//! spec is the publisher's `api.md`):
//! ```c
//! extern "C" __global__ void <entry>(
//!     unsigned long long* start,   // in/out: the work cell, 2 + nthread x u64
//!     unsigned long long count,    // thread-dimension steps in this launch
//!     const unsigned char* data,   // the job data blob (device) — carries the box
//!     unsigned long long data_len,
//!     unsigned char* out,          // output byte-stream buffer (device)
//!     unsigned int* out_count,     // atomic BYTE cursor
//!     unsigned int out_cap);       // capacity in BYTES
//! ```
//! v5's change from v3 is the *shape* of the work space. v3 addressed work by one
//! flat 64-bit index; v5 makes it a **box** of independent per-axis ranges, carried
//! in the job blob, walked as a mixed-radix odometer. Real search spaces are boxes
//! whose axes are not powers of two — flattening one wastes the padding (a Coldcard
//! pad bound of 5e6 inside a 2^32 field leaves each slab 99.88% dead) and overflows
//! 64 bits once axes stack. The cell carries a per-axis cursor instead of an index,
//! and the resume watermark counts *steps of this launch* rather than naming an
//! absolute position.
//!
//! `start[1]` declares how many leading axes this tiler enumerates, so a cubin built
//! to walk a different split rejects the job instead of computing garbage. The axes
//! behind that split are loop dimensions: a thread walks that whole sub-box itself,
//! which is what keeps an expensive stage (a master PBKDF2) amortised across it.
//!
//! Two older ABIs are still supported for jobs already dispatched against them
//! (api.md §6): v3 (a `[base, resume]` cell, 64-bit index) and pre-v3 (`start`
//! passed by value, `out_count`/`out_cap` counted in fixed-size records). The
//! manifest's `format` field selects between them; see [`Abi`]. v4 — a flat index of
//! arbitrary limb width — is *not* among them: it was superseded before any job was
//! dispatched, so nothing speaks it and decryptd refuses it outright.

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
    /// v5 (`format == 5`): the work space is a box of per-axis ranges carried in the
    /// job blob, and `start` points at a `[resume, nthread, pos…]` cell. `thread_fields`
    /// is the manifest's `thread_fields` — the leading axes this tiler enumerates, one
    /// work item per cell — and must equal the count the cubin was built for or it
    /// rejects the job outright.
    V5 { thread_fields: u32 },
}

/// Upper bound on a box's axis count. The deepest space in the spec is ten axes (an
/// L10 brute, one per character); this only stops a malformed blob from making us
/// allocate an absurd work cell.
const MAX_BOX_FIELDS: usize = 64;

/// Written to the resume slot by a cubin whose baked `nthread` disagrees with the one
/// we declared (api.md §4.5). Unambiguous because a launch's `count` is always below
/// it, so it can never be confused with a buffer-full watermark.
const ABI_REJECT: u64 = u64::MAX;

/// The work space a v5 job searches: one inclusive `[start, end]` range per axis, read
/// from the `CPB2` job blob (api.md §2). Axis order is least-significant-first, so
/// axis 0 cycles fastest.
///
/// This is the one thing decryptd looks at inside the blob. It stays payload-agnostic
/// otherwise — it needs the box only to walk the odometer, never to interpret a hit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkBox {
    /// `(start, end)` inclusive per axis, all `nfields` of them.
    axes: Vec<(u64, u64)>,
    /// Leading axes the tiler enumerates (api.md §1.3). The rest are loop dimensions:
    /// every thread walks that whole sub-box internally, so they never reach the cell.
    nthread: usize,
}

const CPB2_MAGIC: u32 = 0x3242_5043; // "CPB2"

impl WorkBox {
    /// Parse the box out of a `CPB2` blob, taking `nthread` from the manifest.
    fn parse(data: &[u8], thread_fields: u32) -> Result<WorkBox, String> {
        let rd32 = |off: usize| -> Result<u32, String> {
            data.get(off..off + 4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .ok_or_else(|| format!("job blob is {} bytes, too short for CPB2", data.len()))
        };
        let magic = rd32(0)?;
        if magic != CPB2_MAGIC {
            return Err(format!(
                "job blob magic {magic:#010x} is not CPB2 ({CPB2_MAGIC:#010x}) — a v5 \
                 manifest needs a v5 blob"
            ));
        }
        let nfields = rd32(20)? as usize;
        if nfields == 0 || nfields > MAX_BOX_FIELDS {
            return Err(format!(
                "job blob declares {nfields} box axes (expected 1..={MAX_BOX_FIELDS})"
            ));
        }
        let need = 24 + 16 * nfields;
        if data.len() < need {
            return Err(format!(
                "job blob is {} bytes, too short for {nfields} box axes ({need} needed)",
                data.len()
            ));
        }
        let at = |off: usize| u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
        let mut axes = Vec::with_capacity(nfields);
        for i in 0..nfields {
            let (lo, hi) = (at(24 + 8 * i), at(24 + 8 * (nfields + i)));
            // api.md §1.5: `end < u64::MAX` keeps the radix from overflowing to 0, and
            // `start <= end` keeps it non-empty. Both are the publisher's job; a blob
            // that breaks either would make the odometer silently wrong.
            if lo > hi {
                return Err(format!("box axis {i} is empty: start {lo} > end {hi}"));
            }
            if hi == u64::MAX {
                return Err(format!("box axis {i} ends at u64::MAX; it must be split"));
            }
            axes.push((lo, hi));
        }
        // An absent `thread_fields` means the whole box is thread dimensions — the
        // shape a core with no amortising inner loop bakes.
        let nthread = if thread_fields == 0 {
            nfields
        } else {
            thread_fields as usize
        };
        if nthread > nfields {
            return Err(format!(
                "manifest thread_fields {nthread} exceeds the blob's {nfields} box axes"
            ));
        }
        Ok(WorkBox { axes, nthread })
    }

    /// Span of axis `i` — its radix in the odometer.
    fn radix(&self, i: usize) -> u128 {
        let (lo, hi) = self.axes[i];
        u128::from(hi - lo) + 1
    }

    /// Total thread-dimension cells in the box: the product of the thread axes' radices,
    /// which is the unit the platform's fragment ranges and the manifest's `tile` count
    /// in. Loop dimensions are not included — a thread walks those internally.
    fn cells(&self) -> Result<u128, String> {
        (0..self.nthread).try_fold(1u128, |acc, i| {
            acc.checked_mul(self.radix(i))
                .ok_or_else(|| "box has more than 2^128 thread-dimension cells".to_string())
        })
    }

    /// Largest radix among the thread axes — the term api.md §1.5 bounds a launch's
    /// `count` against, so the kernel's `(r_i - 1) + carry` cannot overflow.
    fn max_radix(&self) -> u128 {
        (0..self.nthread).map(|i| self.radix(i)).max().unwrap_or(1)
    }

    /// The odometer cursor `linear` steps into the box: one absolute axis value per
    /// thread dimension, least-significant axis first (api.md §1.4 — the kernel then
    /// advances this by `tid` the same digit-wise way).
    fn cursor(&self, linear: u128) -> Vec<u64> {
        let mut rest = linear;
        (0..self.nthread)
            .map(|i| {
                let (lo, _) = self.axes[i];
                let r = self.radix(i);
                let digit = rest % r;
                rest /= r;
                lo + digit as u64
            })
            .collect()
    }
}

/// Where a run currently sits, in whatever terms its ABI addresses work. This — not
/// the [`Abi`] tag — is what shapes the work cell and the watermark read-back, so the
/// two can never disagree.
enum Cursor {
    /// v3 / legacy: a flat 64-bit work-item index.
    Index(u64),
    /// v5: the odometer position over the box's thread dimensions.
    Box(Vec<u64>),
}

impl Cursor {
    /// Build the `start` work cell for a launch of `count` items (api.md §1.2/§6). v5
    /// is `[resume=count, nthread, pos…]`; v3 is `[base, resume=base+count]`. (The
    /// legacy ABI has no cell — it passes the base index by value and never calls this.)
    ///
    /// Under both cell ABIs the resume slot is *seeded* by the publisher and only ever
    /// lowered by the kernel — v5 seeds it with the launch's step count, v3 with the
    /// absolute end of the launch's range.
    fn work_cell(&self, count: u64) -> Vec<u64> {
        match self {
            Cursor::Index(base) => vec![*base, base.saturating_add(count)],
            Cursor::Box(pos) => {
                let mut cell = Vec::with_capacity(2 + pos.len());
                cell.push(count);
                cell.push(pos.len() as u64);
                cell.extend_from_slice(pos);
                cell
            }
        }
    }

    /// Byte range of the resume slot within the cell: v5 puts it first, v3 second.
    fn resume_slot(&self) -> std::ops::Range<usize> {
        match self {
            Cursor::Box(_) => 0..8,
            Cursor::Index(_) => 8..16,
        }
    }

    /// How many steps of a launch of `count` items completed, read back from the resume
    /// slot (api.md §4). `Err` means the cubin refused the job — its baked shape
    /// disagrees with what the manifest declared, so every launch would fail the same.
    ///
    /// v5's watermark is *launch-relative* ("steps done"), which is what removed v4's
    /// one unresolvable corner: there is no absolute end value that might not fit the
    /// slot, and "ran to the end" is just `count`, not a distinguished sentinel.
    fn steps_done(&self, count: u64, resume: u64) -> Result<u64, ()> {
        match self {
            Cursor::Box(_) if resume == ABI_REJECT => Err(()),
            Cursor::Box(_) => Ok(resume.min(count)),
            // v3's watermark is the absolute index it stopped at. Clamp: a kernel that
            // leaves the cell untouched (or writes nonsense) must not rewind the range
            // or carry it past the end.
            Cursor::Index(base) => Ok(resume.clamp(*base, base.saturating_add(count)) - base),
        }
    }
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
/// records under [`Abi::Legacy`] and bytes under v3/v5. Either way, under-sizing it is
/// safe — no result is ever dropped — but the ABIs recover differently:
///
/// * **v3/v5** use the kernel's resume watermark (api.md §4). A launch whose buffer
///   fills reports how far it got; everything before that is complete, so the next
///   launch simply restarts there. No work is redone.
/// * **legacy** has no watermark, so an overflowing launch's buffer holds an arbitrary
///   subset (whichever threads won the atomic race) and is unusable. The per-launch
///   item count adapts to the observed match density, targeting ~7/8 fill, and an
///   overflowing launch is recomputed at the corrected size — its work is redone, but
///   the density estimate makes overflow rare (typically once, to calibrate).
///
/// Under either ABI, a single item that alone overflows `out_cap` is a hard error
/// (the cap is too small for the job).
///
/// `start`/`end_incl` are the platform's fragment range. Under v5 they are a linear
/// offset into the box's thread-dimension cells — the same unit `tile` counts in — and
/// decryptd converts each position to an odometer cursor via the box in `data`. Under
/// v3 and legacy they are the flat work-item index directly, and a range past
/// `u64::MAX` is rejected up front rather than silently truncated. They are `u128`
/// either way because a box's cell count routinely exceeds 64 bits (api.md §1.1).
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
    // Bytes of output buffer per unit of `out_cap`: v3/v5 count bytes directly, legacy
    // counts fixed-size records.
    let cap_unit = match abi {
        Abi::Legacy { record_size } => {
            if record_size == 0 {
                return Err("manifest record_size is 0".into());
            }
            record_size as usize
        }
        Abi::V3 | Abi::V5 { .. } => 1,
    };
    // v5 reads the work space out of the job blob; every other ABI addresses work by a
    // flat 64-bit index, so its range must fit one.
    let work_box = match abi {
        Abi::V5 { thread_fields } => Some(WorkBox::parse(data, thread_fields)?),
        Abi::Legacy { .. } | Abi::V3 => {
            if end_incl > u128::from(u64::MAX) {
                return Err(format!(
                    "fragment reaches work index {end_incl}, past the 64-bit index this \
                     job's ABI addresses"
                ));
            }
            None
        }
    };
    // The box bounds the run: the publisher may hand out a fragment that overshoots it
    // (api.md §1.4 lets the odometer's carry stop over-sized launches), but there is no
    // point launching for cells that don't exist.
    let end_incl = match &work_box {
        Some(b) => end_incl.min(b.cells()?.saturating_sub(1)),
        None => end_incl,
    };

    let func = gpu.function(entry)?;
    let d_data = DeviceBuf::from_slice(data)?;
    let d_out = DeviceBuf::alloc(cap_unit * out_cap as usize)?;
    let d_count = DeviceBuf::alloc(4)?;
    // v3/v5 only: the work cell that `start` points at. Its presence is also what
    // marks this run as framed below.
    let d_start = match (abi, &work_box) {
        (Abi::Legacy { .. }, _) => None,
        // `[resume, nthread, pos…]`, one cursor word per thread dimension. Sized from
        // the parsed box, not the manifest field, which may have left `nthread` implied.
        (Abi::V5 { .. }, Some(b)) => Some(DeviceBuf::alloc(8 * (2 + b.nthread))?),
        _ => Some(DeviceBuf::alloc(16)?), // v3: `[base, resume]`
    };

    let total = end_incl.saturating_sub(start).saturating_add(1);
    // api.md §1.5 bounds a launch: `count < u64::MAX` keeps the rejection sentinel
    // unambiguous, and `count <= u64::MAX - max(r_i)` keeps the kernel's per-thread
    // `(r_i - 1) + carry` from overflowing. No real tile comes near either, but a
    // manifest is publisher input — clamp it rather than trust it.
    let tile_cap = match &work_box {
        Some(b) => u64::try_from(u128::from(u64::MAX) - b.max_radix()).unwrap_or(u64::MAX - 1),
        None => u64::MAX - 1,
    };
    let tile = tile.clamp(1, tile_cap.max(1));
    let mut results = Vec::new();
    let mut done = 0u128;
    let mut cur = start;
    // Wall-clock budget for this fragment, minus any time spent parked in `gate`
    // (a paused worker must not time out). Checked once per tile below.
    let started = Instant::now();
    let mut paused = Duration::ZERO;
    // Items per launch. Under v3/v5 this stays a full tile: a launch that fills the
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

        // Where this launch starts, in the terms its ABI addresses work: an odometer
        // cursor over the box's thread dimensions under v5, a flat index otherwise.
        let cursor = match &work_box {
            Some(b) => Cursor::Box(b.cursor(cur)),
            None => Cursor::Index(cur as u64), // range checked to fit a u64 above
        };
        let count = ((end_incl - cur).saturating_add(1))
            .min(u128::from(launch))
            .max(1) as u64;
        let end_x = cur + u128::from(count);
        d_count.memset0()?;
        if let Some(ds) = &d_start {
            // v3/v5: zero the output buffer — the zero tail *is* the stream terminator,
            // and it also masks the previous launch's bytes past an overflow point —
            // and seed the work cell (the kernel only ever lowers the resume slot).
            d_out.memset0()?;
            let cell: Vec<u8> = cursor
                .work_cell(count)
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            ds.htod(&cell)?;
        }
        // v3/v5 pass a device pointer to the work cell; legacy passes the base index by
        // value (its cursor is always an `Index`). Both are one u64-sized slot.
        let mut a_start = match (&d_start, &cursor) {
            (Some(ds), _) => ds.ptr,
            (None, Cursor::Index(base)) => *base,
            (None, Cursor::Box(_)) => 0,
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
            // ---- v3/v5: the resume watermark says exactly how far this launch got.
            // v5 puts it in `start[0]`, v3 in `start[1]`.
            let mut cell = vec![0u8; ds.len];
            ds.dtoh(&mut cell)?;
            let resume = u64::from_le_bytes(cell[cursor.resume_slot()].try_into().unwrap());
            // A rejection means the cubin was built for a different `nthread` than the
            // manifest declares. Every launch of this job would fail the same way, so
            // stop rather than burn the whole fragment discovering it repeatedly.
            let steps = cursor.steps_done(count, resume).map_err(|()| {
                "cubin refused the job (ABI-reject sentinel): it was not built for the \
                 manifest's thread_fields"
                    .to_string()
            })?;
            if steps == 0 {
                // The buffer filled before the launch's very first item could be
                // recorded, so it made no progress. Retry the same position with fewer
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
            let next = cur + u128::from(steps);
            // `raw` is the byte cursor, which overshoots `out_cap` when the buffer
            // filled; the framing walk finds the true end of the written prefix.
            //
            // Everything before the watermark is complete, so we keep the whole prefix
            // and restart at `next`. Threads race, so a filled buffer can also hold
            // records for steps at or past the watermark, and re-running from it emits
            // those a second time. That's inherent to the ABI (api.md §4 prescribes
            // exactly this loop) and decryptd can't dedup — it doesn't know the payload
            // layout, so it can't tell which cell a record belongs to. The consumer
            // tolerates a repeated record; a dropped one it could not.
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

    /// Build a `CPB2` blob header carrying `axes` (api.md §2). The param and charset
    /// regions that follow it are what decryptd never looks at.
    fn cpb2(axes: &[(u64, u64)]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&CPB2_MAGIC.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // algo
        b.extend_from_slice(&0u32.to_le_bytes()); // pwlen
        b.extend_from_slice(&0u32.to_le_bytes()); // clen
        b.extend_from_slice(&0u32.to_le_bytes()); // nparam
        b.extend_from_slice(&(axes.len() as u32).to_le_bytes()); // nfields
        for (lo, _) in axes {
            b.extend_from_slice(&lo.to_le_bytes());
        }
        for (_, hi) in axes {
            b.extend_from_slice(&hi.to_le_bytes());
        }
        b
    }

    /// The box comes out of the blob, and its thread axes are what the fragment range
    /// and `tile` count in — loop dimensions are walked inside a thread and must not
    /// inflate the cell count (api.md §1.3).
    #[test]
    fn work_box_parses_and_counts_thread_cells() {
        // A `[pad, ssr, tod]` box with a `warm` loop dimension behind it.
        let axes = [(0, 4_999_999), (0, 254), (0, 86_399), (0, 224)];
        let blob = cpb2(&axes);

        let b = WorkBox::parse(&blob, 3).expect("three thread dimensions");
        assert_eq!(b.cells().unwrap(), 5_000_000 * 255 * 86_400);
        // The loop axis is parsed but excluded from the tiled space.
        assert_eq!(b.axes.len(), 4);

        // Every axis a thread dimension: the loop axis now multiplies in.
        let b = WorkBox::parse(&blob, 4).expect("four thread dimensions");
        assert_eq!(b.cells().unwrap(), 5_000_000 * 255 * 86_400 * 225);

        // An absent `thread_fields` (0) means exactly that — the whole box.
        assert_eq!(WorkBox::parse(&blob, 0).unwrap(), b);

        // `thread_fields` past the blob's axis count is a mismatch, not a clamp.
        assert!(WorkBox::parse(&blob, 5).is_err());
    }

    /// The odometer: axis 0 cycles fastest, and each digit is an absolute axis value
    /// (`start_i` + digit), not an offset (api.md §1.4/§2).
    #[test]
    fn work_box_cursor_walks_least_significant_first() {
        // Radices 10 and 7, both offset off zero so absolute values are visible.
        let b = WorkBox::parse(&cpb2(&[(100, 109), (5, 11)]), 2).unwrap();
        assert_eq!(b.cells().unwrap(), 70);
        assert_eq!(b.cursor(0), vec![100, 5]);
        assert_eq!(b.cursor(1), vec![101, 5]); // axis 0 cycles fastest
        assert_eq!(b.cursor(9), vec![109, 5]);
        assert_eq!(b.cursor(10), vec![100, 6]); // carry into axis 1
        assert_eq!(b.cursor(69), vec![109, 11]); // last cell in the box

        // A box far past 2^64 is addressed digit-wise with no flat index formed —
        // an L10 brute over 95 glyphs is 95^10 = 2^66.4 cells.
        let l10 = WorkBox::parse(&cpb2(&[(0, 94); 10]), 10).unwrap();
        assert_eq!(l10.cells().unwrap(), 95u128.pow(10));
        assert!(l10.cells().unwrap() > u128::from(u64::MAX));
        assert_eq!(l10.cursor(95 * 95), vec![0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
    }

    /// A blob that isn't CPB2, or whose axes break api.md §1.5, must be a handled error
    /// — the odometer would be silently wrong rather than loudly broken.
    #[test]
    fn work_box_rejects_malformed_blobs() {
        assert!(WorkBox::parse(&[], 1).is_err(), "empty blob");
        let mut wrong_magic = cpb2(&[(0, 9)]);
        wrong_magic[0] ^= 0xff;
        assert!(WorkBox::parse(&wrong_magic, 1).is_err(), "not CPB2");
        // Truncated before the end[] array.
        let full = cpb2(&[(0, 9), (0, 9)]);
        assert!(WorkBox::parse(&full[..full.len() - 8], 2).is_err(), "short");
        // start > end is an empty axis; end == u64::MAX overflows the radix to 0.
        assert!(WorkBox::parse(&cpb2(&[(9, 0)]), 1).is_err(), "empty axis");
        assert!(
            WorkBox::parse(&cpb2(&[(0, u64::MAX)]), 1).is_err(),
            "full u64"
        );
    }

    /// The v5 work cell (api.md §1.2) is `[resume, nthread, pos…]` with `resume` seeded
    /// to the launch's step count — v3's is `[base, base+count]`, and the legacy ABI has
    /// no cell at all.
    #[test]
    fn work_cell_layout() {
        let b = WorkBox::parse(&cpb2(&[(100, 109), (5, 11)]), 2).unwrap();
        let cur = Cursor::Box(b.cursor(12));
        assert_eq!(cur.work_cell(8), vec![8, 2, 102, 6]);
        assert_eq!(cur.resume_slot(), 0..8, "v5 puts resume first");

        let idx = Cursor::Index(100);
        assert_eq!(
            idx.work_cell(8),
            vec![100, 108],
            "v3 seeds the absolute end of its range",
        );
        assert_eq!(idx.resume_slot(), 8..16, "v3 puts resume second");
    }

    /// The watermark says how many steps of the launch completed. v5's is
    /// launch-relative, v3's is the absolute index it stopped at; both reduce to the
    /// same "steps done" the caller advances its cursor by.
    #[test]
    fn resume_watermark_reports_steps_done() {
        let cur = Cursor::Box(vec![100]);
        // Untouched seed: the launch ran to the end.
        assert_eq!(cur.steps_done(8, 8), Ok(8));
        // Buffer filled after 5 steps; everything before that is recorded.
        assert_eq!(cur.steps_done(8, 5), Ok(5));
        // Filled before the first step — no progress; the caller retries smaller.
        assert_eq!(cur.steps_done(8, 0), Ok(0));
        // Garbage above the seed can't come from a legal atomicMin; clamp to the count
        // rather than carrying the cursor past the launch.
        assert_eq!(cur.steps_done(8, 99), Ok(8));
        // `nthread` mismatch: the cubin refused the job outright.
        assert_eq!(cur.steps_done(8, u64::MAX), Err(()));

        // v3's slot holds an absolute index, so the same outcomes come out shifted.
        let idx = Cursor::Index(100);
        assert_eq!(idx.steps_done(8, 108), Ok(8));
        assert_eq!(idx.steps_done(8, 105), Ok(5));
        assert_eq!(idx.steps_done(8, 100), Ok(0));
        assert_eq!(idx.steps_done(8, 3), Ok(0), "below base: clamped");
        assert_eq!(idx.steps_done(8, u64::MAX), Ok(8));
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
