pub const target = @import("target.zig");
pub const is_device = target.is_device;
pub const kernel_callconv = target.kernel_callconv;

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

pub const Reducer = host.Reducer;

pub const THREADS = @import("device/config.zig").THREADS;
pub const EPT = @import("device/config.zig").EPT;
pub const TILE = @import("device/config.zig").TILE;

pub const Matmul = host.Matmul;

pub fn GlobalPtr(comptime T: type) type {
    return [*]T;
}

pub fn ConstGlobalPtr(comptime T: type) type {
    return [*]const T;
}

pub const TensorAccess = enum {
    read,
    write,
    read_write,
};

pub fn TensorPtr(comptime T: type, comptime access: TensorAccess) type {
    return switch (access) {
        .read => ConstGlobalPtr(T),
        .write, .read_write => GlobalPtr(T),
    };
}

pub fn Tensor(
    comptime T: type,
    comptime rank: u32,
    comptime access: TensorAccess,
) type {
    return struct {
        ptr: TensorPtr(T, access),
        shape: [rank]u32,
        stride: [rank]u32,
    };
}
