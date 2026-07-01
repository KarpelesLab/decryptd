//! Minimal CUDA Driver-API wrapper for the generic launch path. decryptd knows
//! nothing about the kernel's job — it uploads an opaque data blob, launches a
//! kernel with the fixed ABI below over a range, and reads back the output records.
//!
//! Kernel ABI (the contract every decryptd cubin entry point implements):
//! ```c
//! extern "C" __global__ void <entry>(
//!     unsigned long long start,    // first work-item index
//!     unsigned long long count,    // items in this launch
//!     const unsigned char* data,   // the opaque job data blob (device)
//!     unsigned long long data_len,
//!     unsigned char* out,          // output record buffer (device)
//!     unsigned int* out_count,     // atomically-incremented record counter
//!     unsigned int out_cap);       // capacity in records
//! ```

use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;

type CuResult = i32;
type CuDevice = i32;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuDeviceptr = u64;

#[allow(non_snake_case)]
unsafe extern "C" {
    fn cuInit(flags: u32) -> CuResult;
    fn cuDeviceGet(device: *mut CuDevice, ordinal: i32) -> CuResult;
    fn cuDeviceGetAttribute(pi: *mut i32, attrib: i32, dev: CuDevice) -> CuResult;
    fn cuDeviceGetName(name: *mut c_char, len: i32, dev: CuDevice) -> CuResult;
    fn cuCtxCreate_v2(pctx: *mut CuContext, flags: u32, dev: CuDevice) -> CuResult;
    fn cuCtxDestroy_v2(ctx: CuContext) -> CuResult;
    fn cuModuleLoadData(module: *mut CuModule, image: *const c_void) -> CuResult;
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

/// An initialized CUDA context with a module loaded.
pub struct Gpu {
    #[allow(dead_code)]
    ctx: CuContext,
    module: CuModule,
    dev: CuDevice,
}

impl Gpu {
    /// Init device 0 and load the best cubin for it. Callers pass `(arch, bytes)`
    /// pairs highest-arch-first, where arch is CC `X.Y` encoded as `X*10+Y`.
    ///
    /// Cubins newer than the device are skipped rather than tried: an old driver
    /// (e.g. 550.x / CUDA 12.4) doesn't cleanly reject a cubin for an architecture
    /// it has never heard of — `cuModuleLoadData` faults with SIGILL *inside*
    /// libcuda. So we query the GPU's compute capability first and never hand the
    /// driver anything above it. Same-major-lower cubins that still don't load
    /// (a known arch the driver rejects) fall through to the next candidate.
    pub fn load_first(cubins: &[(u32, Vec<u8>)]) -> Result<Gpu, String> {
        unsafe {
            check(cuInit(0), "cuInit")?;
            let mut dev: CuDevice = 0;
            check(cuDeviceGet(&mut dev, 0), "cuDeviceGet")?;

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
                    return Ok(Gpu { ctx, module, dev });
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

/// Run the generic kernel `entry` over `[start, end]` (inclusive), tiling by `tile`
/// items per launch. `data` is the opaque job blob (uploaded once). Returns the raw
/// output records (`out_count * record_size` bytes, concatenated across tiles), and
/// reports per-tile progress via `progress(done, total)`.
#[allow(clippy::too_many_arguments)]
pub fn run_job(
    gpu: &Gpu,
    entry: &str,
    data: &[u8],
    start: u64,
    end_incl: u64,
    record_size: u32,
    out_cap: u32,
    block: u32,
    tile: u64,
    mut progress: impl FnMut(u64, u64),
) -> Result<Vec<u8>, String> {
    let func = gpu.function(entry)?;
    let d_data = DeviceBuf::from_slice(data)?;
    let d_out = DeviceBuf::alloc(record_size as usize * out_cap as usize)?;
    let d_count = DeviceBuf::alloc(4)?;

    let total = end_incl.saturating_sub(start).saturating_add(1);
    let tile = tile.max(1);
    let mut results = Vec::new();
    let mut done = 0u64;
    let mut cur = start;
    while cur <= end_incl {
        let count = ((end_incl - cur).saturating_add(1)).min(tile);
        d_count.memset0()?;
        let (mut a_start, mut a_count) = (cur, count);
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
        let grid = count.div_ceil(block as u64) as u32;
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
        let n = u32::from_le_bytes(cb).min(out_cap);
        if n > 0 {
            let mut recs = vec![0u8; n as usize * record_size as usize];
            // Read only the populated prefix of the output buffer.
            let mut tmp = DeviceBufView {
                ptr: d_out.ptr,
                len: recs.len(),
            };
            tmp.dtoh(&mut recs)?;
            results.extend_from_slice(&recs);
        }
        done += count;
        progress(done.min(total), total);
        cur = cur.saturating_add(count);
        if count == 0 {
            break;
        }
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
