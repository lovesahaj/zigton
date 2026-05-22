// --- context.zig
const std = @import("std");
const cuda = @import("cuda");

const zt = @import("zigton");

// --- main.zig
const ptx = @embedFile("vector_add.ptx");

// init: std.process.Init
pub fn main() !void {
    const n: u32 = 1024;
    const bytes = n * @sizeOf(f32);

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
    var module: cuda.CUmodule = undefined;
    try zt.check(cuda.cuModuleLoadData(&module, ptx.ptr));
    defer _ = cuda.cuModuleUnload(module);

    // get the vector add kernel
    var func: cuda.CUfunction = undefined;
    try zt.check(cuda.cuModuleGetFunction(&func, module, "vector_add"));

    // Allocate memory block on the GPU VRAM
    var dx: cuda.CUdeviceptr = undefined;
    var dy: cuda.CUdeviceptr = undefined;
    var dz: cuda.CUdeviceptr = undefined;

    try zt.check(cuda.cuMemAlloc_v2(&dx, bytes));
    defer _ = cuda.cuMemFree_v2(dx);

    try zt.check(cuda.cuMemAlloc_v2(&dy, bytes));
    defer _ = cuda.cuMemFree_v2(dy);

    try zt.check(cuda.cuMemAlloc_v2(&dz, bytes));
    defer _ = cuda.cuMemFree_v2(dz);

    // uploading the array data to the allocated device chunks
    try zt.check(cuda.cuMemcpyHtoD_v2(dx, &x, bytes));
    try zt.check(cuda.cuMemcpyHtoD_v2(dy, &y, bytes));

    // Marashalling arguments matching the expected pointer structures
    var arg_x = dx;
    var arg_y = dy;
    var arg_z = dz;
    var arg_n = n;

    var args = [_]?*anyopaque{
        &arg_x,
        &arg_y,
        &arg_z,
        &arg_n,
    };

    const block_x: c_uint = 256;
    const grid_x: c_uint = (n + block_x - 1) / block_x;

    try zt.check(cuda.cuLaunchKernel(func, grid_x, 1, 1, // Grid Dimensions
        block_x, 1, 1, // Block Dimensions
        0, null, // Shared memory, Stream context
        @ptrCast(&args), null));

    // Await complete task cluster evalution
    try ctx.sync();

    // Pull results directly back into local memory array
    try zt.check(cuda.cuMemcpyDtoH_v2(&z, dz, bytes));

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
