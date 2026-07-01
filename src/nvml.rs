//! Minimal NVML (`libnvidia-ml`) binding for live GPU temperature / power in the
//! tray. NVML ships with the NVIDIA driver we already require, but we load it at
//! *runtime* through `libloading` rather than linking it: nothing to add to the
//! build, and a machine without NVML (or a call the driver doesn't support) just
//! yields no readout instead of failing.
//!
//! NVML enumerates every physical GPU and does *not* honor `CUDA_VISIBLE_DEVICES`,
//! so callers look devices up by their PCI bus id (see [`crate::cuda::pci_bus_id`])
//! — the one identifier both APIs agree on.

use std::ffi::{CString, c_char, c_int, c_uint, c_void};

use libloading::{Library, Symbol};

/// `nvmlReturn_t` — 0 is `NVML_SUCCESS`.
type Ret = c_int;
/// `nvmlDevice_t` — an opaque handle.
type Device = *mut c_void;

// NVML entry points we use. All are thread-safe per the NVML docs.
type InitFn = unsafe extern "C" fn() -> Ret;
type ShutdownFn = unsafe extern "C" fn() -> Ret;
type HandleByPciFn = unsafe extern "C" fn(*const c_char, *mut Device) -> Ret;
type TemperatureFn = unsafe extern "C" fn(Device, c_int, *mut c_uint) -> Ret;
type PowerFn = unsafe extern "C" fn(Device, *mut c_uint) -> Ret;

/// `NVML_TEMPERATURE_GPU` — the die sensor.
const NVML_TEMPERATURE_GPU: c_int = 0;

/// A live temperature / power reading for one GPU. Either field is `None` when the
/// driver couldn't supply it.
#[derive(Default)]
pub struct Telemetry {
    /// Die temperature in °C.
    pub temp_c: Option<u32>,
    /// Board power draw in watts.
    pub power_w: Option<f64>,
}

/// A loaded, initialized NVML. The resolved function pointers stay valid for as
/// long as `_lib` is held, so this owns the `Library`.
pub struct Nvml {
    handle_by_pci: HandleByPciFn,
    temperature: TemperatureFn,
    power: PowerFn,
    shutdown: ShutdownFn,
    _lib: Library,
}

// The fields are bare `extern "C"` fn pointers and a `Library` (both Send + Sync);
// NVML itself is thread-safe, so the handle is fine to share across threads.
unsafe impl Send for Nvml {}
unsafe impl Sync for Nvml {}

impl Nvml {
    /// Load and initialize NVML, or `None` if the library is missing / init fails.
    pub fn load() -> Option<Nvml> {
        unsafe {
            let lib = Library::new("libnvidia-ml.so.1")
                .or_else(|_| Library::new("libnvidia-ml.so"))
                .or_else(|_| Library::new("nvml.dll"))
                .ok()?;
            let init: Symbol<InitFn> = lib.get(b"nvmlInit_v2\0").ok()?;
            if init() != 0 {
                return None;
            }
            let handle_by_pci = *lib
                .get::<HandleByPciFn>(b"nvmlDeviceGetHandleByPciBusId_v2\0")
                .ok()?;
            let temperature = *lib
                .get::<TemperatureFn>(b"nvmlDeviceGetTemperature\0")
                .ok()?;
            let power = *lib.get::<PowerFn>(b"nvmlDeviceGetPowerUsage\0").ok()?;
            let shutdown = *lib.get::<ShutdownFn>(b"nvmlShutdown\0").ok()?;
            Some(Nvml {
                handle_by_pci,
                temperature,
                power,
                shutdown,
                _lib: lib,
            })
        }
    }

    /// Temperature + power for the GPU at `pci_bus_id`. Best-effort: a missing
    /// device or unsupported query leaves the corresponding field `None`.
    pub fn telemetry(&self, pci_bus_id: &str) -> Telemetry {
        let mut out = Telemetry::default();
        let Ok(pci) = CString::new(pci_bus_id) else {
            return out;
        };
        unsafe {
            let mut dev: Device = std::ptr::null_mut();
            if (self.handle_by_pci)(pci.as_ptr(), &mut dev) != 0 {
                return out;
            }
            let mut temp: c_uint = 0;
            if (self.temperature)(dev, NVML_TEMPERATURE_GPU, &mut temp) == 0 {
                out.temp_c = Some(temp);
            }
            let mut milliwatts: c_uint = 0;
            if (self.power)(dev, &mut milliwatts) == 0 {
                out.power_w = Some(milliwatts as f64 / 1000.0);
            }
        }
        out
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        unsafe { (self.shutdown)() };
    }
}
