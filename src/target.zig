const std = @import("std");
const builtin = @import("builtin");

pub const is_device = builtin.target.cpu.arch == .nvptx64;

pub const kernel_callconv: std.builtin.CallingConvention = if (is_device) .kernel else .c;
