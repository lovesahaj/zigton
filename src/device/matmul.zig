const utils = @import("utils.zig");
const tensor = @import("tensor.zig");
const sharedtile = @import("sharedtile.zig");

pub fn MatmulThreadMap(
    comptime BM: u32,
    comptime BN: u32,
    comptime TM: u32,
    comptime TN: u32,
) type {
    return struct {
        pub const THREADS_M = BM / TM;
        pub const THREADS_N = BN / TN;

        thread_m: u32,
        thread_n: u32,
        local_tid: u32,
        row_base: u32,
        col_base: u32,

        pub fn init(pid_m: u32, pid_n: u32) @This() {
            const thread_n = utils.threadId(0);
            const thread_m = utils.threadId(1);

            return .{
                .thread_m = thread_m,
                .thread_n = thread_n,
                .local_tid = thread_m * THREADS_N + thread_n,
                .row_base = pid_m * BM + thread_m * TM,
                .col_base = pid_n * BN + thread_n * TN,
            };
        }
    };
}

pub fn AccumulatorTile(comptime T: type, comptime TM: u32, comptime TN: u32) type {
    return struct {
        data: [TM * TN]T,

        pub fn zero() @This() {
            return .{ .data = @splat(0) };
        }

        pub fn addOuter(self: *@This(), a: [TM]T, b: [TN]T) void {
            inline for (0..TM) |mi| {
                inline for (0..TN) |ni| {
                    self.data[mi * TN + ni] += a[mi] * b[ni];
                }
            }
        }
    };
}
