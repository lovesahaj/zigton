const Context = @import("context.zig").Context;
const Kernel = @import("kernel.zig").Kernel;
const DeviceBuffer = @import("buffer.zig").DeviceBuffer;
const cdiv = @import("utils.zig").cdiv;
const kernelArgs = @import("args.zig").kernelArgs;


pub const Module = @import("module.zig").Module;
const config = @import("../device/config.zig");

pub const Reducer = struct {
    block_sum: Kernel,

    const Self = @This();

    pub fn init(module: Module) !void {

    }
};

/// Reduces `input[0..n]` to one f32. May overwrite `input`.
pub fn reduceSumF32(
    ctx: *Context,
    block_sum: Kernel,
    input: DeviceBuffer(f32),
    n: u32,
) !f32 {
    if (n == 0) return 0.0;

    const da = input;

    var db = try DeviceBuffer(f32).init(n);
    defer db.deinit();

    var current_count = n;
    var current_is_a = true;

    while (current_count > 1) {
        const in_ptr = if (current_is_a) da.ptr else db.ptr;
        const out_ptr = if (current_is_a) db.ptr else da.ptr;

        const number_of_blocks = try cdiv(current_count, config.TILE); // grid_x
        const number_of_threads = config.THREADS; // block_x

        const args = kernelArgs(.{
            in_ptr,
            out_ptr,
            current_count,
        });

        try block_sum.launch(.{
            .grid = .{ .x = number_of_blocks },
            .block = .{ .x = number_of_threads },
        }, args);

        try ctx.sync();
        current_count = number_of_blocks;
        current_is_a = !current_is_a;
    }

    var out: [1]f32 = undefined;

    if (current_is_a) {
        try da.copyToHost(out[0..]);
    } else {
        try db.copyToHost(out[0..]);
    }

    return out[0];
}
