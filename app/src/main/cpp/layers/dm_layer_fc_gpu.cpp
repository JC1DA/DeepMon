//
// Created by JC1DA on 4/11/17.
//

#include <dm_blob.hpp>
#include <vector>
#include <dm_common.hpp>
#include <layers/dm_layer_fc.hpp>
#include <clblast_c.h>
#include <clblast.h>
#include <dm.hpp>

using namespace std;
using namespace deepmon;

namespace deepmon {
    /*DM_Blob* DM_Layer_Fc::ForwardGpu(vector<DM_Blob *> blobs) {
        if(blobs.size() != 1) {
            LOGE("[%s] has more than 1 input", this->name.c_str());
            return NULL;
        }

        DM_Blob *input = blobs[0];

        int batches = input->get_shape_at(0);

        DM_Blob *output = new DM_Blob(vector<uint32_t> {
                input->get_shape_at(0), output_shapes[0]
        }, ENVIRONMENT_GPU, this->precision, NULL);

        int m = batches;
        int n = num_neurons;
        int k = input_size;

        cl_mem data_in = input->get_gpu_data();
        cl_mem data_out = output->get_gpu_data();

        cl_event event;
        CLBlastStatusCode status;

        cl_command_queue queue = DeepMon::Get().GetGpuExecutionEngine().GetCurrentQueue();

        if(precision == PRECISION_32) {
            status = CLBlastSgemm(CLBlastLayoutRowMajor,
                                  CLBlastTransposeNo, CLBlastTransposeYes,
                                  m, n, k,
                                  1.0f,
                                  data_in, 0, k,
                                  filters->get_gpu_data(), 0, k,
                                  0,
                                  data_out, 0, n,
                                  &queue, &event);
        } else if(precision == PRECISION_16) {
            status = CLBlastHgemm(CLBlastLayoutRowMajor,
                                  CLBlastTransposeNo, CLBlastTransposeYes,
                                  m, n, k,
                                  1.0f,
                                  data_in, 0, k,
                                  filters->get_gpu_data(), 0, k,
                                  0,
                                  data_out, 0, n,
                                  &queue, &event);
        }

        if (status == CLBlastSuccess) {
            clWaitForEvents(1, &event);
            clReleaseEvent(event);
        } else {
            LOGE("[%s]: Weights-Gemm failed with status %d", this->name.c_str(), status);
            output->set_corrupted(true);
        }

        //process biases
        if (status == CLBlastSuccess && biases != NULL) {
            float * biases_multiplier = new float[n];
            for (int i = 0; i < n; i++)
                biases_multiplier[i] = 1;

            DM_Blob *biases_multiplier_blob = NULL;
            if (biases != NULL) {
                biases_multiplier_blob = new DM_Blob(vector<uint32_t>{(uint32_t)n}, ENVIRONMENT_GPU,
                                                     this->precision, biases_multiplier);
            }

            cl_mem biases_multiplier_blob_gpu_data = biases_multiplier_blob->get_gpu_data();
            if(precision == PRECISION_32) {
                status = CLBlastSgemm(CLBlastLayoutRowMajor,
                                      CLBlastTransposeNo, CLBlastTransposeNo,
                                      batches, n, 1,
                                      1.0f,
                                      biases_multiplier_blob_gpu_data, 0, 1,
                                      biases->get_gpu_data(), 0, n,
                                      1.0,
                                      data_out, 0, n,
                                      &queue, &event);
            } else if(precision == PRECISION_16) {
                status = CLBlastHgemm(CLBlastLayoutRowMajor,
                                      CLBlastTransposeNo, CLBlastTransposeNo,
                                      batches, n, 1,
                                      1.0f,
                                      biases_multiplier_blob_gpu_data, 0, 1,
                                      biases->get_gpu_data(), 0, n,
                                      1.0,
                                      data_out, 0, n,
                                      &queue, &event);
            }

            if (status == CLBlastSuccess) {
                clWaitForEvents(1, &event);
                clReleaseEvent(event);
            } else {
                LOGE("[%s]: Biases-Gemm failed with status %d", this->name.c_str(), status);
                output->set_corrupted(true);
            }

            delete biases_multiplier_blob;
            delete biases_multiplier;
        }

        return output;
    }*/

    DM_Blob* DM_Layer_Fc::ForwardGpu(vector<DM_Blob *> blobs) {
        if(blobs.size() != 1) {
            LOGE("[%s] has more than 1 input", this->name.c_str());
            return NULL;
        }

        DM_Blob *input = blobs[0];

        int batches = input->get_shape_at(0);

        DM_Blob *output = new DM_Blob(vector<uint32_t> {
                input->get_shape_at(0), output_shapes[0]
        }, ENVIRONMENT_GPU, this->precision, NULL);

        int m = batches;
        int n = num_neurons;
        int k = input_size;

        cl_mem data_in = input->get_gpu_data();
        cl_mem data_out = output->get_gpu_data();

        cl_int err = CL_SUCCESS;
        cl_command_queue current_queue = DeepMon::Get().GetGpuExecutionEngine().GetCurrentQueue();
        cl_kernel kernel = DeepMon::Get().GetGpuExecutionEngine().GetKernel(precision, KERNEL_DM_FC_BASE);

        for(int batch_idx = 0 ; batch_idx < batches ; batch_idx++) {
            int offset_idx = batch_idx;
            int i = 0;

            err = clSetKernelArg(kernel, i++, sizeof(cl_int), &offset_idx);
            err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &data_in);
            int input_size = input->get_size() / batches;
            err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &input_size);
            cl_mem weights_data = this->filters->get_gpu_data();
            err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &weights_data);
            cl_mem biases_data = this->biases->get_gpu_data();
            err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &biases_data);
            err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &data_out);
            int output_size = output->get_size() / batches;
            err |= clSetKernelArg(kernel, i++, sizeof(cl_int), &output_size);

            SAMPLE_CHECK_ERRORS(err);
            if(err != CL_SUCCESS) {
                output->set_corrupted(true);
                return output;
            }

            size_t wgs[1] = {(size_t)output_size};

            err = clEnqueueNDRangeKernel(
                    current_queue,
                    kernel,
                    1,
                    0,
                    wgs,
                    0,
                    0, 0, 0
            );
            SAMPLE_CHECK_ERRORS(err);
            if(err != CL_SUCCESS) {
                output->set_corrupted(true);
                return output;
            }
        }

        err = clFinish(current_queue);
        SAMPLE_CHECK_ERRORS(err);
        if(err != CL_SUCCESS) {
            output->set_corrupted(true);
            return output;
        }

        return output;
    }
}