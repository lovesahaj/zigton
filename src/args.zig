// pub fn KernelArgs(comptime Args: type) type {
//     const fields = @typeInfo(Args).@"struct".fields;
//
//     return struct {
//         values: Args,
//         ptrs: [fields.len]?*anyopaque,
//
//         const Self = @This();
//
//         pub fn init(values: Args) Self {
//             var self: Self = .{
//                 .values = values,
//                 .ptrs = undefined,
//             };
//
//             inline for (fields, 0..) |field, i| {
//                 const field_name = field.name;
//                 self.ptrs[i] = @ptrCast(&@field(self.values, field_name));
//             }
//
//             return self;
//         }
//
//         pub fn ptr(self: *Self) [*]?*anyopaque {
//             return &self.ptrs;
//         }
//     };
// }
//
// pub fn kernelArgs(values: anytype) KernelArgs(@TypeOf(values)) {
//     return KernelArgs(@TypeOf(values)).init(values);
// }

pub fn KernelArgs4(
    comptime A0: type,
    comptime A1: type,
    comptime A2: type,
    comptime A3: type,
) type {
    return struct {
        a0: A0,
        a1: A1,
        a2: A2,
        a3: A3,
        ptrs: [4]?*anyopaque = undefined,

        const Self = @This();

        pub fn init(a0: A0, a1: A1, a2: A2, a3: A3) Self {
             return .{
                .a0 = a0,
                .a1 = a1,
                .a2 = a2,
                .a3 = a3,
            };
        }

        pub fn ptr(self: *Self) [*]?*anyopaque {
            self.ptrs = .{
                @ptrCast(&self.a0),
                @ptrCast(&self.a1),
                @ptrCast(&self.a2),
                @ptrCast(&self.a3),
            };
            return &self.ptrs;
        }
    };
}

pub fn kernelArgs4(
    a0: anytype,
    a1: anytype,
    a2: anytype,
    a3: anytype,
) KernelArgs4(@TypeOf(a0), @TypeOf(a1), @TypeOf(a2), @TypeOf(a3)) {
    return KernelArgs4(
        @TypeOf(a0),
        @TypeOf(a1),
        @TypeOf(a2),
        @TypeOf(a3),
    ).init(a0, a1, a2, a3);
}
