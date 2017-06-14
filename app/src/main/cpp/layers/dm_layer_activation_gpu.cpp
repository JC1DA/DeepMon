//
// Created by JC1DA on 6/7/17.
//

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