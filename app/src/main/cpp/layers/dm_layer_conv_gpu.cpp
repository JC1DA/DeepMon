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
#include <dm.hpp>

using namespace deepmon;
namespace deepmon {
    void DM_Layer_Conv::CAFFE_LAYOUT_im2col_gpu(DM_Blob *input, DM_Blob *output) {
        int batches = input->get_shape_at(0);

        for(int b = 0 ; b < batches ; b++) {
            uint32_t input_offset = b * input->get_shape_at(1) * input->get_shape_at(2) * input->get_shape_at(3);
            uint32_t output_offset = b * output->get_shape_at(1) * output->get_shape_at(2) * output->get_shape_at(3);

            DeepMon::Get().GetGpuExecutionEngine().ExecuteIm2Col(
                    MEMORY_LAYOUT_CAFFE, this->precision, input, input_offset,
                    filter_h, filter_w, stride_h, stride_w,
                    pad_left, pad_top, pad_right, pad_bottom,
                    dilation_h, dilation_w, output_h, output_w,
                    output, output_offset);
        }
    }

    void DM_Layer_Conv::CAFFE_LAYOUT_conv_gpu(DM_Blob *input, DM_Blob *output) {

        /*
         * FIXME: Currently force batch size to be 1
         */

        /*std::vector<uint32_t> im2col_shapes{
                input->get_shape_at(CAFFE_BLOB_INOUT_BATCH_IDX),
                num_channels * filter_h * filter_w,
                output->get_shape_at(CAFFE_BLOB_INOUT_HEIGHT_IDX),
                output->get_shape_at(CAFFE_BLOB_INOUT_WIDTH_IDX)
        };*/
        std::vector<uint32_t> im2col_shapes{
                num_channels * filter_h * filter_w,
                output->get_shape_at(CAFFE_BLOB_INOUT_HEIGHT_IDX) * output->get_shape_at(CAFFE_BLOB_INOUT_WIDTH_IDX)
        };

        DM_Blob *im2col_blob = new DM_Blob(im2col_shapes, ENVIRONMENT_GPU, this->precision,
                                           NULL);

        CAFFE_LAYOUT_im2col_gpu(input, im2col_blob);

        int input_offset = im2col_blob->get_shape_at(1) * im2col_blob->get_shape_at(2) *
                           im2col_blob->get_shape_at(3);
        int output_offset =
                output->get_shape_at(1) * output->get_shape_at(2) * output->get_shape_at(3);

        int m = filters->get_shape_at(CAFFE_BLOB_FILTER_NUM_FILTERS);
        int k = im2col_blob->get_shape_at(1);
        int n = output->get_shape_at(CAFFE_BLOB_INOUT_HEIGHT_IDX) *
                output->get_shape_at(CAFFE_BLOB_FILTER_WIDTH);

        float *biases_multiplier = NULL;
        if (biases != NULL) {
            biases_multiplier = new float[n];
            for (int i = 0; i < n; i++)
                biases_multiplier[i] = 1;
        }

        DM_Blob *biases_multiplier_blob = NULL;
        if (biases != NULL) {
            biases_multiplier_blob = new DM_Blob(vector<uint32_t>({(uint32_t)n}), ENVIRONMENT_GPU,
                                                 this->precision, biases_multiplier);
        }

        /*
         * FIXME: use ACL matrix multiplication to compute final results
         */
        CLGEMM gemm;
        gemm.configure(im2col_blob->get_CLTensor(), this->filters->get_CLTensor(), NULL, output->get_CLTensor(), 1.0f, 0.0f);

        
        delete im2col_blob;

        if(biases != NULL) {
            delete biases_multiplier;
            delete biases_multiplier_blob;
        }
    }

    void DM_Layer_Conv::DM_LAYOUT_conv_gpu(DM_Blob *input, DM_Blob *output) {
        cl_int err = CL_SUCCESS;
        cl_command_queue current_queue = DeepMon::Get().GetGpuExecutionEngine().GetCurrentQueue();

        cl_kernel kernel = DeepMon::Get().GetGpuExecutionEngine().GetKernel(precision, KERNEL_DM_CONV_LOCAL);

        cl_mem cl_input = input->get_gpu_data();
        cl_mem cl_output = output->get_gpu_data();

        for(int idx = 0 ; idx < input->get_shape_at(0) ; idx++) {
            int offset_idx = idx;
            int i = 0;
            err  = clSetKernelArg(kernel, i++, sizeof(cl_int), &offset_idx);
            err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_input);
            err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->input_w);
            err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->input_h);
            err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->num_channels);
            cl_mem filters_data = this->filters->get_gpu_data();
            err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &filters_data);
            cl_mem biases_data  = this->biases->get_gpu_data();
            err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &biases_data);
            err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->filter_w);
            err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->filter_h);
            err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->num_filters);
            err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->stride_w);
            err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->stride_h);
            err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->pad_left);
            err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->pad_top);
            err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_output);
            err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &output_w);
            err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &output_h);
            SAMPLE_CHECK_ERRORS(err);
            if(err != CL_SUCCESS) {
                output->set_corrupted(true);
                return;
            }

            size_t lgs[2] = {(size_t)128, (size_t)1};

            int wgs_1 = ((output_h * output_w / lgs[0]) + ((output_h * output_w % lgs[0] == 0) ? 0 : 1)) * lgs[0];
            size_t wgs[2] = {(size_t)wgs_1, (size_t)num_filters};

            err = clEnqueueNDRangeKernel(
                    current_queue,
                    kernel,
                    2,
                    0,
                    wgs,
                    lgs,
                    0, 0, 0
            );
            SAMPLE_CHECK_ERRORS(err);
            if(err != CL_SUCCESS) {
                output->set_corrupted(true);
                return;
            }
        }

        err = clFinish(current_queue);
        SAMPLE_CHECK_ERRORS(err);
        if(err != CL_SUCCESS) {
            output->set_corrupted(true);
            return;
        }
    }

    DM_Blob* DM_Layer_Conv::do_conv_gpu(DM_Blob *input) {
        //need to add batch_size
        DM_Blob *output = new DM_Blob(vector<uint32_t> {
                input->get_shape_at(0), output_shapes[0], output_shapes[1], output_shapes[2]
        }, ENVIRONMENT_GPU, this->precision, NULL);

        if (mem_layout == MEMORY_LAYOUT_CAFFE) {
            CAFFE_LAYOUT_conv_gpu(input, output);
        } else if(mem_layout == MEMORY_LAYOUT_DM) {
            DM_LAYOUT_conv_gpu(input, output);
        }

        return output;
    }
}
