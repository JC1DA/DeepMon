//
// Created by JC1DA on 3/8/17.
//

#include <dm_log.hpp>
#include "dm_execution_engine_cpu.hpp"
#include "dm_execution_engine_gpu.hpp"

namespace deepmon {
    void CAFFE_LAYOUT_im2col_cpu(const float *data_im, const int channels, const int height,
                                 const int width, const int kernel_h, const int kernel_w,
                                 const int pad_h, const int pad_w, const int stride_h,
                                 const int stride_w, const int dilation_h,
                                 const int dilation_w, float *data_col) {
        const int output_h = (height + 2 * pad_h
                              - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
        const int output_w =
                (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
        const int channel_size = height * width;
        for (int channel = channels; channel--; data_im += channel_size) {
            for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
                for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                    int input_row = -pad_h + kernel_row * dilation_h;
                    for (int output_rows = output_h; output_rows; output_rows--) {
                        if (!(input_row >= 0 && input_row < height)) {
                            for (int output_cols = output_w; output_cols; output_cols--) {
                                *(data_col++) = 0;
                            }
                        } else {
                            int input_col = -pad_w + kernel_col * dilation_w;
                            for (int output_col = output_w; output_col; output_col--) {
                                if ((input_col >= 0 && input_col< width)) {
                                    *(data_col++) = data_im[input_row * width + input_col];
                                } else {
                                    *(data_col++) = 0;
                                }
                                input_col += stride_w;
                            }
                        }
                        input_row += stride_h;
                    }
                }
            }
        }
    }

    void DM_LAYOUT_im2col_cpu(const float *data_im, const int channels, \
                              const int height, const int width, const int kernel_h, const int kernel_w, \
                              const int pad_top, const int pad_left, const int pad_bottom, const int pad_right, \
                              const int stride_h, const int stride_w, \
                              const int dilation_h, const int dilation_w, float *data_col) {
        const int output_h = (height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
        const int output_w = (width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

        for(int output_row = 0 ; output_row < output_h ; output_row++) {
            for(int output_col = 0 ; output_col < output_w ; output_col++) {
                for(int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
                    int input_row = output_row * stride_h - pad_top + kernel_row * dilation_h;
                    if(input_row < 0 || input_row >= height) {
                        //set 0 to all data
                        int num_items = kernel_w * channels;
                        memset(data_col, 0, num_items * sizeof(float));
                        data_col += num_items;
                    } else {
                        for(int kernel_col = 0 ; kernel_col < kernel_w ; kernel_col++) {
                            int input_col = output_col * stride_w - pad_left + kernel_col * dilation_h;
                            if(input_col < 0 || input_col >= width) {
                                int num_items = channels;
                                memset(data_col, 0, num_items * sizeof(float));
                                data_col += num_items;
                            } else {
                                int input_idx = (input_row * width + input_col) * channels;
                                memcpy(data_col, &data_im[input_idx], channels * sizeof(float));
                                data_col += channels;
                            }
                        }
                    }
                }
            }
        }
    }

    void DM_Execution_Engine_CPU::do_im2col(ENVIRONMENT_TYPE evn_type, MEMORY_LAYOUT mem_layout,
                                            DM_Blob *input, DM_Blob *output,
                                            std::vector<int> filters_sizes,
                                            std::vector<int> strides, std::vector<int> pads,
                                            std::vector<int> dilations) {
        if(evn_type != ENVIRONMENT_CPU) {
            output->set_corrupted(true);
            LOGE("Incorrect Environment");
            return;
        }

        /*
         * To get better performance, we should move these variables to layer class
         */
        int batches, channels, height, width, input_im_size, output_im_size, kernel_h, kernel_w;
        if(input->get_shapes().size() == 4) {
            batches = input->get_shape_at(0);
            if(mem_layout == MEMORY_LAYOUT_CAFFE) {
                channels = input->get_shape_at(1);
                height = input->get_shape_at(2);
                width = input->get_shape_at(3);
                kernel_h = filters_sizes.at(2);
                kernel_w = filters_sizes.at(3);
            } else if(mem_layout == MEMORY_LAYOUT_DM) {
                channels = input->get_shape_at(3);
                height = input->get_shape_at(1);
                width = input->get_shape_at(2);
                kernel_h = filters_sizes.at(1);
                kernel_w = filters_sizes.at(2);
            }
            input_im_size = input->get_shape_at(1) * input->get_shape_at(2) * input->get_shape_at(3);
            output_im_size = output->get_shape_at(1) * output->get_shape_at(2) * output->get_shape_at(3);
        } else {
            batches = 1;
            if(mem_layout == MEMORY_LAYOUT_CAFFE) {
                channels = input->get_shape_at(0);
                height = input->get_shape_at(1);
                width = input->get_shape_at(2);
                kernel_h = filters_sizes.at(2);
                kernel_w = filters_sizes.at(3);
            } else if(mem_layout == MEMORY_LAYOUT_DM) {
                channels = input->get_shape_at(2);
                height = input->get_shape_at(0);
                width = input->get_shape_at(1);
                kernel_h = filters_sizes.at(1);
                kernel_w = filters_sizes.at(2);
            }
            input_im_size = input->get_shape_at(0) * input->get_shape_at(1) * input->get_shape_at(2);
            output_im_size = output->get_shape_at(0) * output->get_shape_at(1) * output->get_shape_at(2);
        }

        for(int b = 0 ; b < batches ; b++) {
            float *data_im = input->get_cpu_data() + b * input_im_size;
            float *output_im = output->get_cpu_data() + b * output_im_size;
            if(mem_layout == MEMORY_LAYOUT_CAFFE) {
                CAFFE_LAYOUT_im2col_cpu(data_im, channels, height, width, kernel_h, kernel_w,\
                                        pads.at(0), pads.at(1), strides.at(0), strides.at(1), \
                                        dilations.at(0), dilations.at(1), output_im);
            } else {
                DM_LAYOUT_im2col_cpu(data_im, channels, height, width, kernel_h, kernel_w,\
                                        pads.at(0), pads.at(1), pads.at(2), pads.at(3), strides.at(0), strides.at(1), \
                                        dilations.at(0), dilations.at(1), output_im);
            }
        }
    }
}