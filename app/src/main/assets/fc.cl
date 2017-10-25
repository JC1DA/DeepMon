kernel void fc_base(
	const int offset_idx,
    global const real *input_frame,
    const int input_size,
    global const real *layer_W,
    global const real *layer_bias,
    global real *output_frame,
    const int output_size
) {
    for(int n = get_global_id(0); n < output_size ; n += get_global_size(0)) {
        real result = 0.0f;

        int idx_remaining = input_size;

        __global real *input_ptr = input_frame + offset_idx * input_size;
        __global real *filter_ptr = layer_W + n * input_size;

        while(idx_remaining >= VWM) {
            realM tmp1 = vloadM(*input_ptr);
            realM tmp2 = vloadM(*filter_ptr);
            result += dot(tmp1,tmp2);

            input_ptr += VWM;
            filter_ptr += VWM;
            idx_remaining -= VWM;
        }

        while(idx_remaining > 0) {
            result += (*input_ptr) * (*filter_ptr);

            input_ptr++;
            filter_ptr++;

            idx_remaining -= 1;
        }

        output_frame[offset_idx * output_size + n] = result + layer_bias[n];
    }
}