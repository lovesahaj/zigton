const std = @import("std");

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
        }
    }
}
