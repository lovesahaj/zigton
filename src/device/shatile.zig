const utils = @import("utils.zig");

pub fn SharedTile(comptime T: type, comptime n: u32) type {
    return struct {
        data: *addrspace(.shared) [n]T,

        const Self = @This();

        pub fn init(data: *addrspace(.shared) [n]T) Self {
            return .{ .data = data };
        }

        pub fn load(self: Self, i: u32) T {
            return self.data[i];
        }

        pub fn store(self: Self, i: u32, value: T) void {
            self.data[i] = value;
        }
    };
}

pub inline fn sharedTile(
    comptime T: type,
    comptime n: u32,
    comptime tag: anytype,
) SharedTile(T, n) {
    _ = tag;

    // Zig allows static local variabls through nested containers
    // This means we can hide stoarge inside a helper
    const Storage = struct {
        var data: [n]T addrspace(.shared) = undefined;
    };

    return .init(&Storage.data);
}
