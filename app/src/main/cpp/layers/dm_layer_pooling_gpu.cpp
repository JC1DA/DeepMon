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

    void DM_Layer_Pooling::DM_LAYOUT_ForwardGPU(DM_Blob *input, DM_Blob *output) {
        int batches = input->get_shape_at(0);

        cl_int err = CL_SUCCESS;
        cl_command_queue current_queue = DeepMon::Get().GetGpuExecutionEngine().GetCurrentQueue();

        cl_mem cl_input = input->get_gpu_data();
        cl_mem cl_output = output->get_gpu_data();

        cl_kernel kernel;

        if(!type.compare("MAXPOOL"))
            kernel = DeepMon::Get().GetGpuExecutionEngine().GetKernel(this->precision, KERNEL_DM_MAXPOOL);
        else if(!type.compare("AVEPOOL"))
            kernel = DeepMon::Get().GetGpuExecutionEngine().GetKernel(this->precision, KERNEL_DM_AVEPOOL);

        int i = 0;
        err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_input);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->input_w);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->input_h);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->num_channels);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->filter_w);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->filter_h);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->stride_w);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->stride_h);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->pad_left);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &this->pad_top);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_output);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &output_w);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &output_h);
        err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &batches);
        SAMPLE_CHECK_ERRORS(err);
        if(err != CL_SUCCESS) {
            output->set_corrupted(true);
            return;
        }

        size_t wgs[3] = {(size_t)output_w, (size_t)output_h, (size_t)num_channels};

        err = clEnqueueNDRangeKernel(
                current_queue,
                kernel,
                3,
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

    DM_Blob* DM_Layer_Pooling::do_pooling_gpu(DM_Blob *input) {
        DM_Blob *output = new DM_Blob(vector<uint32_t> {
                input->get_shape_at(0), output_shapes[0], output_shapes[1], output_shapes[2]
        }, ENVIRONMENT_GPU, this->precision, NULL);

        if(this->mem_layout == MEMORY_LAYOUT_CAFFE)
            CAFFE_LAYOUT_ForwardGPU(input, output);
        else if(this->mem_layout == MEMORY_LAYOUT_DM) {
            DM_LAYOUT_ForwardGPU(input, output);
        }

        return output;
    }
}

