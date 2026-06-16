const utils = @import("utils.zig");

pub const TensorAccess = enum {
    read,
    write,
    read_write,
};

fn TensorPtr(comptime T: type, comptime access: TensorAccess) type {
    return switch (access) {
        .read => utils.ConstGlobalPtr(T),
        .write, .read_write => utils.GlobalPtr(T),
    };
}

fn canRead(comptime access: TensorAccess) bool {
    return access == .read or access == .read_write;
}

fn canWrite(comptime access: TensorAccess) bool {
    return access == .write or access == .read_write;
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

        const Self = @This();

        pub fn init(
            ptr: TensorPtr(T, access),
            shape: [rank]u32,
            stride: [rank]u32,
        ) Self {
            return .{
                .ptr = ptr,
                .shape = shape,
                .stride = stride,
            };
        }

        pub fn offset(self: Self, idx: [rank]u32) u32 {
            var out: u32 = 0;
            inline for (0..rank) |axis| {
                out += idx[axis] * self.stride[axis];
            }
            return out;
        }

        pub fn inBounds(self: Self, idx: [rank]u32) bool {
            inline for (0..rank) |axis| {
                if (idx[axis] >= self.shape[axis]) return false;
            }

            return true;
        }

        pub fn load(self: Self, idx: [rank]u32, fill: T) T {
            if (!canRead(access)) @compileError("Tensor access mode does not allow load");
            return if (self.inBounds(idx)) self.ptr[self.offset(idx)] else fill;
        }

        pub fn load2D(self: Self, row: u32, col: u32, fill: T) T {
            if (rank != 2) @compileError("load2D requires rank-2 tensors");
            return self.load(.{ row, col }, fill);
        }

        pub fn store(self: Self, idx: [rank]u32, value: T) void {
            if (!canWrite(access)) @compileError("Tensor access mode does not allow store");
            if (self.inBounds(idx)) self.ptr[self.offset(idx)] = value;
        }

        pub fn store2D(self: Self, row: u32, col: u32, value: T) void {
            if (rank != 2) @compileError("store2D requires rank-2 tensors");
            return self.store(.{ row, col }, value);
        }
    };
}
