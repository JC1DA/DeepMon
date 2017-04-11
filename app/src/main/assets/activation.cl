
__kernel void activate_relu(
    const int n,
    __global const real *in,
    __global real *out,
    const real negative_slope
) {
    for(int index = get_global_id(0); index < n; index += get_global_size(0)) {
        out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
    }
}

__kernel void activate_tanh(
    const int n,
    __global const real *in,
    __global real *out
) {
    for(int index = get_global_id(0); index < n; index += get_global_size(0)) {
        out[index] = tanh(in[index]);
    }
}

__kernel void activate_sigmoid(
    const int n,
    __global const real *in,
    __global real *out
) {
    for(int index = get_global_id(0); index < n; index += get_global_size(0)) {
        out[index] = 1.0 / (1.0 + exp(-in[index]));
    }
}