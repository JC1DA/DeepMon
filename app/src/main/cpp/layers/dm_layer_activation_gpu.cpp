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

#include <layers/dm_layer_activation.hpp>
#include <dm.hpp>
#include <clblast_half.h>

namespace deepmon {

    void DM_Layer_Activation::Activation_Leaky_GPU(DM_Blob *input, DM_Blob *output) {
        cl_command_queue queue = DeepMon::Get().GetGpuExecutionEngine().GetCurrentQueue();
        cl_kernel kernel = DeepMon::Get().GetGpuExecutionEngine().GetKernel(precision, KERNEL_ACTIVATE_RELU);

        int i = 0;
        cl_int err = CL_SUCCESS;

        int n = input->get_total_size();
        cl_mem cl_in = input->get_gpu_data();
        cl_mem cl_out = output->get_gpu_data();

        err  = clSetKernelArg(kernel, i++, sizeof(cl_int), &n);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_in);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_out);

        if(precision == PRECISION_32) {
            float zero_data = activation_threshold;
            err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &zero_data);
        } else {
            half threshold = FloatToHalf(activation_threshold);
            err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &threshold);
        }

        SAMPLE_CHECK_ERRORS(err);
        if(err != CL_SUCCESS) {
            output->set_corrupted(true);
            return;
        }

        size_t wgs[1] = {(size_t)(n)};
        err = clEnqueueNDRangeKernel(
                queue,
                kernel,
                1,
                0,
                wgs,
                0,
                0, 0, 0
        );
        err |= clFinish(queue);
        SAMPLE_CHECK_ERRORS(err);
        if(err != CL_SUCCESS) {
            output->set_corrupted(true);
            return;
        }
    }

    void DM_Layer_Activation::Activation_ReLU_GPU(DM_Blob *input, DM_Blob *output) {
        return Activation_Leaky_GPU(input, output);
    }
}