const std = @import("std");
const cuda = @import("cuda");
const zt = @import("zigton");

// --- main.zig
const ptx = @embedFile("vector_add.ptx");

// init: std.process.Init
pub fn main() !void {
    const n: u32 = 1024;

    // Using large stack allocation or arrays
    var x: [n]f32 = undefined;
    var y: [n]f32 = undefined;
    var z: [n]f32 = undefined;

    for (0..n) |i| {
        x[i] = @floatFromInt(i);
        y[i] = @floatFromInt(i * 2);
        z[i] = 0.0;
    }

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    // Load GPU Module -- the ptx file
    var module = try zt.Module.loadData(ptx);
    defer module.deinit();

    // get the vector add kernel
    const vector_add: zt.Kernel = try module.kernel("vector_add");

    // Allocate memory block on the GPU VRAM
    var dx = try zt.DeviceBuffer(f32).alloc(n);
    defer dx.deinit();

    var dy = try zt.DeviceBuffer(f32).alloc(n);
    defer dy.deinit();

    var dz = try zt.DeviceBuffer(f32).alloc(n);
    defer dz.deinit();

    try dx.copyFromHost(x[0..]);
    try dy.copyFromHost(y[0..]);

    var args = zt.kernelArgs(.{ dx.ptr, dy.ptr, dz.ptr, n });
    args.bind();

    const block_x: c_uint = 256;
    const grid_x: c_uint = (n + block_x - 1) / block_x;

    try vector_add.launch(.{
        .grid = .{ .x = grid_x },
        .block = .{ .x = block_x },
        .args = args.ptr(),
    });

    // Await complete task cluster evalution
    try ctx.sync();

    // Pull results directly back into local memory array
    try dz.copyToHost(z[0..]);

    // validate calculations match perfectly
    for (0..n) |i| {
        const expected = x[i] + y[i];
        if (z[i] != expected) {
            std.debug.print("bad at {}: got {}, expected {}\n", .{ i, z[i], expected });
            return error.BadResult;
        }
    }

    std.debug.print("vector_add OK\n", .{});
}
