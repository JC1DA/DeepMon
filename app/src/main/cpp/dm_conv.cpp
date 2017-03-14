//
// Created by JC1DA on 3/10/17.
//

#include <dm.hpp>
#include <dm_execution_engine_gpu.hpp>
#include <cblas.h>
#include "dm_execution_engine_cpu.hpp"
#include "dm_common.hpp"
#include "dm_utilities.hpp"

using namespace deepmon;

namespace deepmon {
    void DM_Execution_Engine_CPU::do_conv(MEMORY_LAYOUT mem_layout, DM_Blob *input, DM_Blob *output,
                                          DM_Blob *filters, DM_Blob *biases,
                                          std::vector<uint32_t> strides, std::vector<uint32_t> pads,
                                          std::vector<uint32_t> dilations) {
        //should move outside for performance
        int batches, channels, height, width, input_im_size, output_im_size, kernel_h, kernel_w;

        if(mem_layout == MEMORY_LAYOUT_CAFFE) {
            batches = input->get_shape_at(CAFFE_BLOB_INOUT_BATCH_IDX);
            channels = input->get_shape_at(CAFFE_BLOB_INOUT_CHANNELS_IDX);
            height = input->get_shape_at(CAFFE_BLOB_INOUT_HEIGHT_IDX);
            width = input->get_shape_at(CAFFE_BLOB_INOUT_WIDTH_IDX);
            kernel_h = filters->get_shapes().at(CAFFE_BLOB_FILTER_HEIGHT);
            kernel_w = filters->get_shapes().at(CAFFE_BLOB_FILTER_WIDTH);
        } else if(mem_layout == MEMORY_LAYOUT_DM) {
            batches = input->get_shape_at(DM_BLOB_INOUT_BATCH_IDX);
            channels = input->get_shape_at(DM_BLOB_INOUT_CHANNELS_IDX);
            height = input->get_shape_at(DM_BLOB_INOUT_HEIGHT_IDX);
            width = input->get_shape_at(DM_BLOB_INOUT_WIDTH_IDX);
            kernel_h = filters->get_shapes().at(DM_BLOB_FILTER_HEIGHT);
            kernel_w = filters->get_shapes().at(DM_BLOB_FILTER_WIDTH);
        }

        //create blob for im2col_data
        std::vector<uint32_t> im2col_shapes;
        if(mem_layout == MEMORY_LAYOUT_CAFFE) {
            im2col_shapes.push_back(batches);
            im2col_shapes.push_back(channels * kernel_h * kernel_w);
            im2col_shapes.push_back(output->get_shape_at(CAFFE_BLOB_INOUT_HEIGHT_IDX));
            im2col_shapes.push_back(output->get_shape_at(CAFFE_BLOB_INOUT_WIDTH_IDX));
        } else if(mem_layout == MEMORY_LAYOUT_DM) {
            im2col_shapes.push_back(batches);
            im2col_shapes.push_back(output->get_shape_at(DM_BLOB_INOUT_HEIGHT_IDX));
            im2col_shapes.push_back(output->get_shape_at(DM_BLOB_INOUT_WIDTH_IDX));
            im2col_shapes.push_back(channels * kernel_h * kernel_w);
        }
        DM_Blob *im2col_blob = new DM_Blob(im2col_shapes, ENVIRONMENT_CPU, PRECISION_32, NULL);

        DeepMon::Get().get_execution_engine(true).do_im2col(ENVIRONMENT_CPU, mem_layout, input, im2col_blob, filters->get_shapes(), strides, pads, dilations);

        //im2col_blob->print_blob();

        //run gemm to get results
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

        for(int b = 0 ; b < batches ; b++) {
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
    }

    void DM_Execution_Engine_GPU::do_conv(MEMORY_LAYOUT mem_layout, DM_Blob *input, DM_Blob *output,
                                          DM_Blob *filters, DM_Blob *biases,
                                          std::vector<uint32_t> strides, std::vector<uint32_t> pads,
                                          std::vector<uint32_t> dilations) {

    }
}