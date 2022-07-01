"""
*Simpson's integrartion*
integrate the function y = f(x)
"""
function simpson(x::Array, y::Array)
    length(y) > 5 || error("y array must have dimension > 5")
    length(x) == length(y) || error("dimension mismatch")

    ret_val = (17*(y[1] + y[end]) + 59 * (y[2] + y[end-1]) + 
    43 * (y[3] + y[end-2]) + 49 * (y[4] + y[end-3]))/48
    for i=5:(length(y)-4)
        ret_val += y[i]
    end
    return (x[2] - x[1]) * ret_val
end;
