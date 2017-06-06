__kernel void caffe_maxpool(
    const int nthreads,
    __global const real* bottom_data,
    const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    __global real* top_data) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, (int)0);
    wstart = max(wstart, (int)0);
    real maxval = -FLT_MAX;
    int maxidx = -1;
    __global const real* bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval;
  }
}

__kernel void caffe_avepool(
    const int nthreads, __global const real* const bottom_data, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, __global real* top_data) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    {
      const int pw = index % pooled_width;
      const int ph = (index / pooled_width) % pooled_height;
      const int c = (index / pooled_width / pooled_height) % channels;
      const int n = index / pooled_width / pooled_height / channels;
      int hstart = ph * stride_h - pad_h;
      int wstart = pw * stride_w - pad_w;
      int hend = min(hstart + kernel_h, height + pad_h);
      int wend = min(wstart + kernel_w, width + pad_w);
      const int pool_size = (hend - hstart) * (wend - wstart);
      hstart = max(hstart, (int)0);
      wstart = max(wstart, (int)0);
      hend = min(hend, height);
      wend = min(wend, width);
      real aveval = 0;
      __global const real* bottom_slice = bottom_data
          + (n * channels + c) * height * width;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          aveval += bottom_slice[h * width + w];
        }
      }
      top_data[index] = aveval / pool_size;
    }
  }
}

__kernel void dm_maxpool(
    __global const real *input_frame,
    const int input_w,
    const int input_h,
    const int num_channels,
    const int filter_w,
    const int filter_h,
    const int stride_w,
    const int stride_h,
    const int pad_w,
    const int pad_h,
    __global real *output_frame,
    const int output_w,
    const int output_h,
    const int batches) {

    int thrId_i = get_global_id(0);
    int thrId_j = get_global_id(1);
    int thrId_k = get_global_id(2);

    int max_i = get_global_size(0);
    int max_j = get_global_size(1);
    int max_k = get_global_size(2);

    int i,j,k;
    int x,y;
    for(k = thrId_k ; k < batches * num_channels ; k += max_k) {
        for(i = thrId_i ; i < output_w ; i += max_i) {
            for(j = thrId_j ; j < output_h ; j += max_j) {
                real max_value = -9999.9f;
                for(x = 0 ; x < filter_w ; x++) {
                    for(y = 0 ; y < filter_h ; y++) {
                        int x_ = i * stride_w + x - pad_w;
                        int y_ = j * stride_h + y - pad_h;
                        int valid = (x_ >= 0 && x_ < input_w && y_ >= 0 && y_ < input_h);
                        real val = (valid != 0) ? input_frame[getIndexFrom3D(input_h, input_w, num_channels, y_, x_, k)] : 0.0f;
                        max_value   = (val > max_value) ? val   : max_value;
                    }
                }
                output_frame[getIndexFrom3D(output_h, output_w, num_channels, j, i, k)] = max_value;
            }
        }
    }
}

__kernel void dm_avepool(
    __global const real *input_frame,
    const int input_w,
    const int input_h,
    const int num_channels,
    const int filter_w,
    const int filter_h,
    const int stride_w,
    const int stride_h,
    const int pad_w,
    const int pad_h,
    __global real *output_frame,
    const int output_w,
    const int output_h,
    const int batches) {

    int thrId_i = get_global_id(0);
    int thrId_j = get_global_id(1);
    int thrId_k = get_global_id(2);

    int max_i = get_global_size(0);
    int max_j = get_global_size(1);
    int max_k = get_global_size(2);

    int i,j,k;
    int x,y;

    for(k = thrId_k ; k < batches * num_channels ; k += max_k) {
        for(i = thrId_i ; i < output_w ; i += max_i) {
            for(j = thrId_j ; j < output_h ; j += max_j) {
                real avg = 0;
                for(x = 0 ; x < filter_w ; x++) {
                    for(y = 0 ; y < filter_h ; y++) {
                        int x_ = i * stride_w + x - pad_w;
                        int y_ = j * stride_h + y - pad_h;
                        int valid = (x_ >= 0 && x_ < input_w && y_ >= 0 && y_ < input_h);
                        avg += (valid != 0) ? input_frame[getIndexFrom3D(input_h, input_w, num_channels, y_, x_, k)] : 0.0f;
                    }
                }
                output_frame[getIndexFrom3D(output_h, output_w, num_channels, j, i, k)] = avg / (filter_w * filter_h);
            }
        }
    }
}