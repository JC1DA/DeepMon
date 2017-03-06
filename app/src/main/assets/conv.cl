
__kernel void dm_conv_base(
    __global const real *input,
    const int n,
    const int input_offset, //w * h * c
    const int input_w,
    const int input_h,
    const int input_c,
    __global const real *filter_Weights,
    __global const real *filter_biases,
    const int filter_w,
    const int filter_h,
    const int filter_c,
    const int filter_n,
    const int stride_w,
    const int stride_h,
    const int pad_left,
    const int pad_right,
    const int pad_top,
    const int pad_bot,
    __global real *output,
    const int output_offset,
    const int output_w,
    const int output_h,
    const int output_c
) {
    int thread_id, x, y, z;
    int output_image_size = output_w * output_h;
    for(thread_id = get_global_id(0); thread_id < n * output_offset; thread_id += get_global_size(0)) {
        int thread_id_n = thread_id / output_offset;
        int thread_id_y = (thread_id % output_offset) / output_image_size;
        int thread_id_x = (thread_id % output_offset) % output_image_size / output_c;
        int thread_id_z = (thread_id % output_offset) % output_image_size % output_c;

        real result = 0;

        for(y = 0 ; y < filter_h ; y++) {
            int global_input_y = thread_id_y * stride_h - pad_top + y;
            for(x = 0 ; x < filter_w ; x++) {
                int global_input_x = thread_id_x * stride_w - pad_left + x;
                if(global_input_x >= 0 && global_input_y >= 0 && global_input_x < input_w && global_input_y < input_h) {
                    int input_start_idx = getIndexFrom4D(n, input_h, input_w, input_c, thread_id_n, global_input_y, global_input_x, 0);
                    int filter_start_idx = getIndexFrom4D(filter_n, filter_h, filter_w, filter_c, thread_id_z, y, x, 0);
                    int remaining = input_c;
                    while(remaining > VWM) {
                        realM input_data = vloadM(input[input_start_idx]);
                        realM filter_data = vloadM(filter_Weights[filter_start_idx]);
                        result += dotM(input_data, filter_data);
                        remaining -= VWM;
                        input_start_idx++;
                        filter_start_idx++;
                    }
                    while(remaining > 0) {
                        result += input[input_start_idx] * filter_Weights[filter_start_idx];
                        remaining -= 1;
                        input_start_idx++;
                        filter_start_idx++;
                    }
                }
            }
        }

        output[getIndexFrom4D(n, output_h, output_w, output_c, thread_id_n, thread_id_y, thread_id_x, thread_id_z)] = result;
    }
}