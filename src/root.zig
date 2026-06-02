const host = @import("host/root.zig");

pub const Context = host.Context;

pub const DeviceBuffer = host.DeviceBuffer;

pub const Module = host.Module;

pub const Dim3 = host.Dim3;

pub const LaunchConfig = host.LaunchConfig;
pub const launch = host.launch;

pub const KernelArgs = host.KernelArgs;
pub const kernelArgs = host.kernelArgs;

pub const Kernel = host.Kernel;

pub const utils = host.utils;
pub const check = host.check;
pub const errorString = host.errorString;

pub const reduce = host.reduce;

pub const THREADS = @import("device/config.zig").THREADS;
pub const EPT = @import("device/config.zig").EPT;
pub const TILE = @import("device/config.zig").TILE;

