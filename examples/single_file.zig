const std = @import("std");
const builtin = @import("builtin");

const is_device = builtin.target.cpu.arch == .nvptx64;
const kernel_callconv = if (is_device) .kernel else .c;

const zt_host = if (is_device) struct {} else @import("zigton");
const zt_device = if (is_device) @import("zigton_device") else struct {
    pub const THREADS: u32 = 1;

    pub fn GlobalPtr(comptime T: type) type {
        return [*]T;
    }

    pub fn linearIndex() u32 {
        return 0;
    }
};

const ptx: [:0]const u8 = if (is_device) "" else @embedFile("single_file_example_ptx");

pub export fn single_file_fill(
    z: zt_device.GlobalPtr(f32),
    value: f32,
    n: u32,
) callconv(kernel_callconv) void {
    if (comptime !is_device) return;

    const i = zt_device.linearIndex();
    if (i < n) z[i] = value;
}

test "single file host launches device kernel" {
    const zt = zt_host;
    const n: u32 = 1024;
    const fill_value: f32 = 7.0;

    var out: [n]f32 = undefined;

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(ptx);
    defer module.deinit();

    const fill = try module.kernel("single_file_fill");

    var dz = try zt.DeviceBuffer(f32).init(n);
    defer dz.deinit();

    const args = zt.kernelArgs(.{ dz.ptr, fill_value, n });

    try fill.launch(.{
        .grid = .{ .x = try zt.utils.cdiv(n, zt.THREADS) },
        .block = .{ .x = zt.THREADS },
    }, args);

    try ctx.sync();
    try dz.copyToHost(out[0..]);

    for (out) |value| {
        try std.testing.expectEqual(fill_value, value);
    }
}
