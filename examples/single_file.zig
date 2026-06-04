const std = @import("std");
const builtin = @import("builtin");

const is_device = builtin.target.cpu.arch == .nvptx64;
// const kernel_callconv = if (is_device) .kernel else .c;
const kernel_callconv: std.builtin.CallingConvention = if (is_device) .kernel else .c;

const zt = if (is_device)
    @import("zigton_device")
else 
    @import("zigton");

const ptx: [:0]const u8 = if (is_device) "" else @embedFile("single_file_example_ptx");

pub export fn single_file_fill(
    z: zt.GlobalPtr(f32),
    value: f32,
    n: u32,
) callconv(kernel_callconv) void {
    if (comptime !is_device) return;

    const i = zt.linearIndex();
    if (i < n) z[i] = value;
}

test "single file host launches device kernel" {
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
