const std = @import("std");

pub fn KernelArgs(comptime Args: type) type {
    const fields = @typeInfo(Args).@"struct".fields;

    return struct {
        storage: [fields.len]Slot align(max_align),
        ptrs: [fields.len]?*anyopaque,

        const Self = @This();
        const Slot = [max_size]u8;

        const max_size = blk: {
            var s: usize = 0;
            for (fields) |f| s = @max(s, @sizeOf(f.type));
            break :blk s;
        };

        const max_align = blk: {
            var a: usize = 1;
            for (fields) |f| a = @max(a, @alignOf(f.type));
            break :blk a;
        };

        pub fn init(values: Args) Self {
            var self: Self = .{ .storage = undefined, .ptrs = undefined };

            inline for (fields, 0..) |field, i| {
                const v: field.type = @field(values, field.name);
                const dst: *field.type = @ptrCast(@alignCast(&self.storage[i]));
                dst.* = v;
            }

            return self;
        }

        pub fn ptr(self: *Self) [*]?*anyopaque {
            inline for (fields, 0..) |_, i| {
                self.ptrs[i] = @ptrCast(&self.storage[i]);
            }
            return &self.ptrs;
        }
    };
}

pub fn kernelArgs(values: anytype) KernelArgs(@TypeOf(values)) {
    return KernelArgs(@TypeOf(values)).init(values);
}
