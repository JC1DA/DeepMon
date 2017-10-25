
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

__kernel void dm_conv_local(
    const int offset_idx,
    __global const real *input,
    const int input_w,
    const int input_h,
    const int input_c,
    __global const real *conv_weight,
    __global const real *bias,
    const int conv_w,
    const int conv_h,
    const int conv_n,
    const int stride_w,
    const int stride_h,
    const int pad_w,
    const int pad_h,
    __global real *output,
    const int output_w,
    const int output_h
) {
    const int threadId_x = get_global_id(0) % output_w;
    const int threadId_y = get_global_id(0) / output_h;
    const int threadId_z = get_global_id(1);

    __local real local_weight[64 * 3 * 3];
    const int K = (input_c < 64) ? input_c : 64;

    real result = 0;

    const int input_offset = offset_idx * input_w * input_h * input_c;
    const int output_offset = offset_idx * output_w * output_h * conv_n;

    const int loop_counts = input_c / K + ((input_c % K) == 0 ? 0 : 1);
    for(int loop_idx = 0 ; loop_idx < loop_counts ; loop_idx++) {
        const int part_c_size = (loop_idx < (input_c / K)) ? K : input_c % K;
        const int size_to_read = part_c_size * conv_w * conv_h;

        __global const real *conv_weight_base = conv_weight + threadId_z * conv_h * conv_w * input_c;
        //read data into local memory
        for(int local_idx = get_local_id(0) ; local_idx < size_to_read ; local_idx += get_local_size(0)) {
            const int c_ = local_idx % part_c_size;
            const int w_ = (local_idx / part_c_size) % conv_w;
            const int h_ = local_idx / part_c_size / conv_w;
            local_weight[local_idx] = conv_weight_base[(h_ * conv_w + w_) * input_c + (loop_idx * K) + c_];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0 ; k < conv_h * conv_w ; k++) {
        	const int y = k / conv_w;
        	const int x = k % conv_w;
        	const int global_input_y = threadId_y * stride_h - pad_h + y;
        	const int global_input_x = threadId_x * stride_w - pad_w + x;

        	int remaining = part_c_size;
        	__global real *GI = input + input_offset + (global_input_y * input_w + global_input_x) * input_c + loop_idx * K;
        	__local  real *LW = local_weight + (y * conv_w + x) * part_c_size;

        	const int need_process = (0 <= global_input_x && \
        								global_input_x < input_w && \
        								0 <= global_input_y && \
        								global_input_y < input_h) ? 1 : 0;

        	while(remaining > VWM) {
        		realM tmp1 = (need_process == 0) ? 0 : vloadM(*LW);
        		realM tmp2 = (need_process == 0) ? 0 : vloadM(*GI);
        		result += dot(tmp1, tmp2);

        		remaining -= VWM;
	            LW += VWM;
	            GI += VWM;
        	}

        	while(remaining > 0) {
        		result += (need_process == 0) ? 0 : ((*LW) * (*GI));
                remaining--;
                LW++;
                GI++;
        	}
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(threadId_x < output_w && threadId_y < output_h)
        output[output_offset + (threadId_y * output_w + threadId_x) * conv_n + threadId_z] = result + bias[threadId_z];
}

__kernel void dm_conv_with_cache(
    __global const real *input,
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
    const int output_w,
    const int output_h,
    const int output_c,
    __global const int *nonCachedIdx_x,
    __global const int *nonCachedIdx_y,
    const int blk_size_x,
    const int blk_size_y
) {
    const int blk_idx = get_global_id(0) / (blk_size_x * blk_size_y);
    const int blk_x = nonCachedIdx_x[blk_idx];
    const int blk_y = nonCachedIdx_y[blk_idx];

    const int idx = get_global_id(0) % (blk_size_x * blk_size_y);

    int threadId_x = blk_x * blk_size_x + idx % blk_size_x;
    int threadId_y = blk_y * blk_size_y + idx / blk_size_y;
    int threadId_z = get_global_id(1);

    real result = 0;

    for(int y = 0 ; y < filter_h ; y++) {
        int global_input_y = threadId_y * stride_h - pad_top + y;
        for(int x = 0 ; x < filter_w ; x++) {
            int global_input_x = threadId_x * stride_w - pad_left + x;
                int remaining = input_c;

                int GI_idx = (global_input_y * input_w + global_input_x) * input_c;
                int GW_idx = ((threadId_z * filter_h + y) * filter_w + x) * filter_c;

                const int need_to_process = (0 <= global_input_x && \
                        								global_input_x < input_w && \
                        								0 <= global_input_y && \
                        								global_input_y < input_h) ? 1 : 0;

                while(remaining > VWM) {
                    realM input_data =  (need_to_process == 0) ? 0 : vloadM(input[GI_idx]);
                    realM filter_data = (need_to_process == 0) ? 0 : vloadM(filter_Weights[GW_idx]);
                    result += dotM(input_data, filter_data);
                    remaining -= VWM;
                    GI_idx += VWM;
                    GW_idx += VWM;
                }
                while(remaining > 0) {
                    result += (need_to_process == 0) ? 0 : (input[GI_idx] * filter_Weights[GW_idx]);
                    remaining--;
                    GI_idx++;
                    GW_idx++;
                }
        }
    }

    output[(threadId_y * output_w + threadId_x) * output_c + threadId_z] = result + filter_biases[threadId_z];
}