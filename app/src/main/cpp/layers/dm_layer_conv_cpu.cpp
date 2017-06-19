/*The MIT License (MIT)
 *
 *Copyright (c) 2013 Thomas Park
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *       of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *       to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *       copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *       The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *THE SOFTWARE.
 */

#include <layers/dm_layer_conv.hpp>
#include <cblas.h>

namespace deepmon {
    void DM_Layer_Conv::CAFFE_LAYOUT_im2col_cpu(DM_Blob *input, DM_Blob *output) {
        float *data_im = input->get_cpu_data();
        float *data_col = output->get_cpu_data();

        int batches = input->get_shape_at(0);

        const int channel_size = input_h * input_w;
        for (int channel = num_channels * batches; channel--; data_im += channel_size) {
            for (int kernel_row = 0; kernel_row < filter_h; kernel_row++) {
                for (int kernel_col = 0; kernel_col < filter_w; kernel_col++) {
                    int input_row = -pad_top + kernel_row * dilation_h;
                    for (int output_rows = output_h; output_rows; output_rows--) {
                        if (!(input_row >= 0 && input_row < input_h)) {
                            for (int output_cols = output_w; output_cols; output_cols--) {
                                *(data_col++) = 0;
                            }
                        } else {
                            int input_col = -pad_left + kernel_col * dilation_w;
                            for (int output_col = output_w; output_col; output_col--) {
                                if ((input_col >= 0 && input_col< input_w)) {
                                    *(data_col++) = data_im[input_row * input_w + input_col];
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

    void DM_Layer_Conv::DM_LAYOUT_im2col_cpu(DM_Blob *input, DM_Blob *output) {
        float *data_im = input->get_cpu_data();
        float *data_col = output->get_cpu_data();

        int batches = input->get_shape_at(0);
        for(int b = 0 ; b < batches ; b++) {
            for(int output_row = 0 ; output_row < output_h ; output_row++) {
                for(int output_col = 0 ; output_col < output_w ; output_col++) {
                    for(int kernel_row = 0; kernel_row < filter_h; kernel_row++) {
                        int input_row = output_row * stride_h - pad_top + kernel_row * dilation_h;
                        if(input_row < 0 || input_row >= input_h) {
                            //set 0 to all data
                            int num_items = filter_w * num_channels;
                            memset(data_col, 0, num_items * sizeof(float));
                            data_col += num_items;
                        } else {
                            for(int kernel_col = 0 ; kernel_col < filter_w ; kernel_col++) {
                                int input_col = output_col * stride_w - pad_left + kernel_col * dilation_h;
                                if(input_col < 0 || input_col >= input_w) {
                                    int num_items = num_channels;
                                    memset(data_col, 0, num_items * sizeof(float));
                                    data_col += num_items;
                                } else {
                                    int input_idx = (input_row * input_w + input_col) * num_channels;
                                    memcpy(data_col, &data_im[input_idx], num_channels * sizeof(float));
                                    data_col += num_channels;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    DM_Blob* DM_Layer_Conv::do_conv_cpu(DM_Blob *input) {
        //need to add batch_size
        DM_Blob* output = new DM_Blob(vector<uint32_t> {
                input->get_shape_at(0), output_shapes[0], output_shapes[1], output_shapes[2]
        }, ENVIRONMENT_CPU, PRECISION_32, NULL);

        std::vector<uint32_t> im2col_shapes;
        if(mem_layout == MEMORY_LAYOUT_CAFFE) {
            im2col_shapes.push_back(input->get_shape_at(CAFFE_BLOB_INOUT_BATCH_IDX));
            im2col_shapes.push_back(num_channels * filter_h * filter_w);
            im2col_shapes.push_back(output->get_shape_at(CAFFE_BLOB_INOUT_HEIGHT_IDX));
            im2col_shapes.push_back(output->get_shape_at(CAFFE_BLOB_INOUT_WIDTH_IDX));
        } else if(mem_layout == MEMORY_LAYOUT_DM) {
            im2col_shapes.push_back(input->get_shape_at(DM_BLOB_INOUT_BATCH_IDX));
            im2col_shapes.push_back(output->get_shape_at(DM_BLOB_INOUT_HEIGHT_IDX));
            im2col_shapes.push_back(output->get_shape_at(DM_BLOB_INOUT_WIDTH_IDX));
            im2col_shapes.push_back(num_channels * filter_h * filter_w);
        }
        DM_Blob *im2col_blob = new DM_Blob(im2col_shapes, ENVIRONMENT_CPU, PRECISION_32, NULL);
        if(mem_layout == MEMORY_LAYOUT_CAFFE) {
            CAFFE_LAYOUT_im2col_cpu(input, im2col_blob);
        } else {
            //DM LAYOUT
            DM_LAYOUT_im2col_cpu(input, im2col_blob);
        }

        int input_offset = im2col_blob->get_shape_at(1) * im2col_blob->get_shape_at(2) * im2col_blob->get_shape_at(3);
        int output_offset = output->get_shape_at(1) * output->get_shape_at(2) * output->get_shape_at(3);

        int m,n,k;
        if(mem_layout == MEMORY_LAYOUT_CAFFE) {
            m = filters->get_shape_at(CAFFE_BLOB_FILTER_NUM_FILTERS);
            k = im2col_blob->get_shape_at(1);
            n = output->get_shape_at(CAFFE_BLOB_INOUT_HEIGHT_IDX) * output->get_shape_at(CAFFE_BLOB_FILTER_WIDTH);
        } else if(mem_layout == MEMORY_LAYOUT_DM) {
            m = filters->get_shape_at(DM_BLOB_FILTER_NUM_FILTERS);
            k = im2col_blob->get_shape_at(3);
            n = output->get_shape_at(DM_BLOB_INOUT_HEIGHT_IDX) * output->get_shape_at(DM_BLOB_INOUT_WIDTH_IDX);
        }

        float *biases_multiplier = NULL;
        if(biases != NULL) {
            biases_multiplier = new float[n];
            for(int i = 0 ; i < n ; i++)
                biases_multiplier[i] = 1;
        }

        for(int b = 0 ; b < input->get_shapes()[0] ; b++) {
            float *data_im = im2col_blob->get_cpu_data() + b * input_offset;
            float *output_im = output->get_cpu_data() + b * output_offset;
            /*matrix_multiplication(filters->get_cpu_data(), n, m, \
                                    data_im, k, n, output_im, tA, tB, 0);*/

            if(mem_layout == MEMORY_LAYOUT_CAFFE) {
                cblas_sgemm(CblasRowMajor,
                            CblasNoTrans,
                            CblasNoTrans,
                            m, n, k,
                            1.0f,
                            filters->get_cpu_data(), k,
                            data_im, n,
                            0, output_im, n);
                if(biases != NULL) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                biases->get_shape_at(0), n, 1,
                                1.0f,
                                biases->get_cpu_data(), 1,
                                biases_multiplier, n,
                                1.0, output_im, n);
                }
            } else if(mem_layout == MEMORY_LAYOUT_DM) {
                cblas_sgemm(CblasRowMajor,
                            CblasNoTrans,
                            CblasTrans,
                            n, m, k,
                            1.0f,
                            data_im, k,
                            filters->get_cpu_data(), k,
                            0, output_im, m);
                if(biases != NULL) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                n, biases->get_shape_at(0), 1,
                                1.0f,
                                biases_multiplier, 1,
                                biases->get_cpu_data(), biases->get_shape_at(0),
                                1.0, output_im, biases->get_shape_at(0));
                }
            }

        }

        if(biases_multiplier != NULL)
            delete biases_multiplier;

        if(!this->persistant_blobs) {
            delete input;
        }

        return output;
    }
}
