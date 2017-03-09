//
// Created by JC1DA on 3/9/17.
//

#include <dm_log.hpp>
#include "dm_utilities.hpp"
#include <sstream>
#include <cblas.h>

using namespace deepmon;

namespace deepmon {
    void test_im2col(DeepMon &dm) {
        int num_filters = 3;
        int num_channels = 2;
        int filter_h = 3;
        int filter_w = 3;
        int input_h = 3;
        int input_w = 3;
        int pad_top = 1;
        int pad_left = 1;
        int pad_bottom = 1;
        int pad_right = 1;

        std::vector<int> caffe_filters_shapes({num_filters, num_channels, filter_h, filter_w});
        std::vector<int> dm_filters_shapes({num_filters, filter_h, filter_w, num_channels});
        std::vector<int> pads({pad_top,pad_left,pad_bottom,pad_right});
        std::vector<int> strides({1,1});
        std::vector<int> dilations({1,1});

        int input_size = num_channels * input_h * input_w;
        float *input_data = new float[input_size];

        for(int i = 0 ; i < input_size ; i++)
            input_data[i] = (float)i;

        DM_Blob *caffe_input = new DM_Blob(
                std::vector<int>({num_channels,input_h,input_w}),
                ENVIRONMENT_CPU,
                PRECISION_32, input_data);

        DM_Blob *dm_input = new DM_Blob(
                std::vector<int>({input_h,input_w,num_channels}),
                ENVIRONMENT_CPU,
                PRECISION_32, input_data);

        int output_h = (input_h + pad_top + pad_bottom - (dilations.at(0) * (filter_h - 1) + 1)) / strides.at(0) + 1;
        int output_w = (input_w + pad_left + pad_right - (dilations.at(1) * (filter_w - 1) + 1)) / strides.at(1) + 1;

        DM_Blob *caffe_output = new DM_Blob(
                std::vector<int>({num_channels * filter_h * filter_w, output_h, output_w}),
                ENVIRONMENT_CPU,
                PRECISION_32, NULL);
        DM_Blob *dm_output = new DM_Blob(
                std::vector<int>({output_h, output_w, num_channels * filter_h * filter_w}),
                ENVIRONMENT_CPU,
                PRECISION_32, NULL);

        /*dm.get_execution_engine(true).do_im2col(ENVIRONMENT_CPU, MEMORY_LAYOUT_CAFFE, caffe_input, caffe_output, caffe_filters_shapes, strides, pads, dilations);
        float *output = caffe_output->get_cpu_data();
        for(int o_h = 0 ; o_h < output_h ; o_h++) {
            for(int o_w = 0 ; o_w < output_w ; o_w++) {
                for(int c = 0 ; c < num_channels ; c++) {
                    for(int y = 0 ; y < filter_h ; y++) {
                        for(int x = 0 ; x < filter_w ; x++) {
                            int input_y_idx = o_h * strides.at(1) - pad_top + y * dilations.at(1);
                            int input_x_idx = o_w * strides.at(0) - pad_left + x * dilations.at(0);
                            int output_idx = (((c * filter_h + y) * filter_w + x) * output_h + o_h) * output_w + o_w;
                            int input_idx = (c * input_h + input_y_idx) * input_w + input_x_idx;
                            if(input_x_idx < 0 || input_x_idx >= input_w || input_y_idx < 0 || input_y_idx >= input_h) {
                                if(output[output_idx] != float(0)) {
                                    LOGE("INCORRECT");
                                    return;
                                }
                            } else {
                                if(output[output_idx] != input_data[input_idx]) {
                                    LOGE("INCORRECT");
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }*/

        dm.get_execution_engine(true).do_im2col(ENVIRONMENT_CPU, MEMORY_LAYOUT_DM, dm_input, dm_output, dm_filters_shapes, strides, pads, dilations);
        float *output = dm_output->get_cpu_data();
        for(int o_h = 0 ; o_h < output_h ; o_h++) {
            for(int o_w = 0 ; o_w < output_w ; o_w++) {
                for(int c = 0 ; c < num_channels ; c++) {
                    for(int y = 0 ; y < filter_h ; y++) {
                        for(int x = 0 ; x < filter_w ; x++) {
                            int input_y_idx = o_h * strides.at(1) - pad_top + y * dilations.at(1);
                            int input_x_idx = o_w * strides.at(0) - pad_left + x * dilations.at(0);
                            int output_idx = (((o_h * output_w + o_w) * filter_h + y) * filter_w + x) * num_channels + c;
                            int input_idx = (input_y_idx * input_w + input_x_idx) * num_channels + c;
                            if(input_x_idx < 0 || input_x_idx >= input_w || input_y_idx < 0 || input_y_idx >= input_h) {
                                if(output[output_idx] != float(0)) {
                                    LOGE("INCORRECT");
                                    return;
                                }
                            } else {
                                if(output[output_idx] != input_data[input_idx]) {
                                    LOGE("INCORRECT");
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }

        print_blob_3d(*dm_output, 2);
    }

    void print_blob_3d(DM_Blob &blob, int dimension) {
        if(blob.get_shapes().size() != 3 || dimension < 0 || dimension > 2)
            return;

        int first_dim = blob.get_shape_at(0);
        int second_dim = blob.get_shape_at(1);
        if(dimension == 0) {
            first_dim = 1;
            second_dim = 2;
        } else if(dimension == 1) {
            first_dim = 0;
            second_dim = 2;
        }

        for(int x = 0 ; x < first_dim ; x++) {
            for(int y = 0 ; y < second_dim ; y++) {
                std::string str("");
                for(int c = 0 ; c < blob.get_shape_at(dimension) ; c++) {
                    int idx;
                    if(dimension == 0) {
                        idx = (c * first_dim + x) * second_dim + y;
                    } else if(dimension == 1) {
                        idx = (x * blob.get_shape_at(dimension) + c) * second_dim + y;
                    } else {
                        idx = (x * second_dim + y) * blob.get_shape_at(dimension) + c;
                    }
                    float data = blob.get_cpu_data()[idx];
                    std::stringstream ss;
                    ss << data;
                    str += ss.str();
                    str += " ";
                }
                LOGD("(dim_1, dim_2) = (%d,%d): %s",x,y,str.c_str());
            }
        }
    }

    void matrix_multiplcation(float *A, int A_width_, int A_height_,
                              float *B, int B_width_, int B_height_,
                              float *AB, bool tA, bool tB, float beta)
    {
        int A_height = tA ? A_width_  : A_height_;
        int A_width  = tA ? A_height_ : A_width_;
        int B_height = tB ? B_width_  : B_height_;
        int B_width  = tB ? B_height_ : B_width_;
        int m = A_height;
        int n = B_width;
        int k = A_width;
        // Error, width and height should match!
        if(A_width != B_height) {
            return;
        }

        int lda = tA ? m : k;
        int ldb = tB ? k : n;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, A, lda, B, ldb, beta, AB, n);
    }

    void test_openblas() {
        int M = 100;
        int N = 10;
        int K = 100;
        float *a = new float[M * N];
        for(int i = 0 ; i < M * N ; i++)
            a[i] = 1.0f;
        float *b = new float[N * K];
        for(int i = 0 ; i < N * K ; i++)
            b[i] = 1.0f;
        float *c = new float[M * K];
        matrix_multiplcation(a,N,M,b,K,N,c,false,false,0);
        for(int i = 0 ; i < M * K ; i++) {
            if(c[i] != N) {
                LOGE("Incorrect");
                return;
            }
        }
    }
}