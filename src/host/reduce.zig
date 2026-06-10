const std = @import("std");

const Context = @import("context.zig").Context;
const Kernel = @import("kernel.zig").Kernel;
const DeviceBuffer = @import("buffer.zig").DeviceBuffer;
const cdiv = @import("utils.zig").cdiv;
const kernelArgs = @import("args.zig").kernelArgs;

const Module = @import("module.zig").Module;
const config = @import("../device/config.zig");

pub const Reducer = struct {
    block_sum: Kernel,
    block_max: Kernel,
    block_mul: Kernel,
    block_min: Kernel,

    const Self = @This();

    pub fn init(module: Module) !Self {
        return .{
            .block_sum = try module.kernel("block_sum"),
            .block_max = try module.kernel("block_max"),
            .block_mul = try module.kernel("block_mul"),
            .block_min = try module.kernel("block_min"),
        };
    }

    /// Reduces `input[0..n]` to one f32. May overwrite `input`.
    pub fn sumF32(
        self: Self,
        ctx: *Context,
        da: DeviceBuffer(f32),
        db: DeviceBuffer(f32),
        n: u32,
    ) !f32 {
        return self.reduceF32(.Add, ctx, da, db, n);
    }

    /// Reduces `input[0..n]` to one f32. May overwrite `input`.
    pub fn maxF32(
        self: Self,
        ctx: *Context,
        da: DeviceBuffer(f32),
        db: DeviceBuffer(f32),
        n: u32,
    ) !f32 {
        return self.reduceF32(.Max, ctx, da, db, n);
    }

    /// Reduces `input[0..n]` to one f32. May overwrite `input`.
    pub fn minF32(
        self: Self,
        ctx: *Context,
        da: DeviceBuffer(f32),
        db: DeviceBuffer(f32),
        n: u32,
    ) !f32 {
        return self.reduceF32(.Min, ctx, da, db, n);
    }

    /// Reduces `input[0..n]` to one f32. May overwrite `input`.
    pub fn prodF32(
        self: Self,
        ctx: *Context,
        da: DeviceBuffer(f32),
        db: DeviceBuffer(f32),
        n: u32,
    ) !f32 {
        return self.reduceF32(.Mul, ctx, da, db, n);
    }

    /// Reduces `input[0..n]` to one f32. May overwrite `input`.
    pub fn reduceF32(
        self: Self,
        comptime op: std.builtin.ReduceOp,
        ctx: *Context,
        da: DeviceBuffer(f32),
        db: DeviceBuffer(f32),
        n: u32,
    ) !f32 {
        const identity = switch (op) {
            .Add => 0.0,
            .Max => -std.math.inf(f32),
            .Min => std.math.inf(f32),
            .Mul => 1.0,
            else => @compileError("unsupported reduction op"),
        };

        if (n == 0) return identity;

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

            const kernel = switch (op) {
                .Add => self.block_sum,
                .Max => self.block_max,
                .Min => self.block_min,
                .Mul => self.block_mul,
                else => @compileError("unsupported reduction op"),
            };

            try kernel.launch(.{
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
};
