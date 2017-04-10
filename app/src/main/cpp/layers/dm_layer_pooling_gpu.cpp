#include <layers/dm_layer_pooling.hpp>
#include <dm_layer_param.hpp>
#include <dm.hpp>

using namespace deepmon;

namespace deepmon {
    void DM_Layer_Pooling::CAFFE_LAYOUT_ForwardGPU(DM_Blob *input, DM_Blob *output) {
        int batches = input->get_shape_at(0);
        int count = output->get_total_size();

        cl_int err = CL_SUCCESS;
        cl_command_queue current_queue = DeepMon::Get().GetGpuExecutionEngine().GetCurrentQueue();

        cl_mem cl_input = input->get_gpu_data();
        cl_mem cl_output = output->get_gpu_data();

        cl_kernel kernel;

        if(!type.compare("MAXPOOL"))
            kernel = DeepMon::Get().GetGpuExecutionEngine().GetKernel(this->precision, KERNEL_CAFFE_MAXPOOL);
        else if(!type.compare("AVEPOOL"))
            kernel = DeepMon::Get().GetGpuExecutionEngine().GetKernel(this->precision, KERNEL_CAFFE_AVEPOOL);

        int i = 0;
        err  = clSetKernelArg(kernel, i++, sizeof(cl_int), &count);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_input);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &batches);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &input->get_shapes()[CAFFE_BLOB_INOUT_CHANNELS_IDX]);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &input->get_shapes()[CAFFE_BLOB_INOUT_HEIGHT_IDX]);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &input->get_shapes()[CAFFE_BLOB_INOUT_WIDTH_IDX]);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &output->get_shapes()[CAFFE_BLOB_INOUT_HEIGHT_IDX]);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &output->get_shapes()[CAFFE_BLOB_INOUT_WIDTH_IDX]);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &filter_h);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &filter_w);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &stride_h);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &stride_w);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &pad_top);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &pad_left);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_output);

        SAMPLE_CHECK_ERRORS(err);
        if(err != CL_SUCCESS) {
            output->set_corrupted(true);
            return;
        }

        size_t wgs[1] = {(size_t)(output_h * output_w)};
        err = clEnqueueNDRangeKernel(
                current_queue,
                kernel,
                1,
                0,
                wgs,
                0,
                0, 0, 0
        );
        err |= clFinish(current_queue);
        SAMPLE_CHECK_ERRORS(err);

        if(err != CL_SUCCESS) {
            output->set_corrupted(true);
            return;
        }
    }
}