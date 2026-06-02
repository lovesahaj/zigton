const std = @import("std");
const cuda = @import("cuda");

pub const MathError = error{
    DivisionByZero,
};

pub fn cdiv(x: anytype, y: anytype) !c_uint {
    const T = @TypeOf(x, y);

    // Notice the lowercase .int and .float here
    switch (@typeInfo(T)) {
        .int => {
            if (y == 0) return MathError.DivisionByZero;

            const quotient = x / y;
            const remainder = x % y;

            const result = if (remainder > 0) quotient + 1 else quotient;
            return @intCast(result);
        },
        .float => {
            if (@abs(y) <= std.math.floatEps(T)) {
                return MathError.DivisionByZero;
            }

            const float_result = @ceil(x / y);
            return @intFromFloat(float_result);
        },
        else => {
            @compileError("cdiv only supports integers and floats");
        },
    }
}

pub const CudaError = error{
    OutOfMemory,
    NotInitialized,
    InvalidValue,
    InvalidPtx,
    NoDevice,
    LaunchFailed,
    Unknown,
};

pub fn check(result: cuda.CUresult) CudaError!void {
    return switch (result) {
        cuda.CUDA_SUCCESS => {},
        cuda.CUDA_ERROR_OUT_OF_MEMORY => error.OutOfMemory,
        cuda.CUDA_ERROR_NOT_INITIALIZED => error.NotInitialized,
        cuda.CUDA_ERROR_INVALID_VALUE => error.InvalidValue,
        cuda.CUDA_ERROR_INVALID_PTX => error.InvalidPtx,
        cuda.CUDA_ERROR_NO_DEVICE => error.NoDevice,
        cuda.CUDA_ERROR_LAUNCH_FAILED => error.LaunchFailed,
        else => error.Unknown,
    };
}

pub fn errorString(result: cuda.CUresult) []const u8 {
    var ptr: [*c]const u8 = null;
    if (cuda.cuGetErrorString(result, &ptr) != cuda.CUDA_SUCCESS) return "unknown";
    if (ptr == null) return "unknown";
    return std.mem.span(ptr); // [*c] coerces to [*:0] for span
}
