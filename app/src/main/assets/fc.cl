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

        int input_idx = 0;
        int filter_idx = n * input_size;

        int idx_remaining = input_size;

        while(idx_remaining >= VWM) {
            realM tmp1 = vloadM(input_frame[input_idx]);
            realM tmp2 = vloadM(layer_W[filter_idx]);
            result += dot(tmp1,tmp2);

            input_idx += VWM;
            filter_idx += VWM;
            idx_remaining -= VWM;
        }

        while(idx_remaining > 0) {
            real tmp1 = input_frame[input_idx];
            real tmp2 = layer_W[filter_idx];
            result += tmp1 * tmp2;

            idx_remaining -= 1;
        }

        output_frame[offset_idx * output_size + n] = result + layer_bias[n];
    }
}