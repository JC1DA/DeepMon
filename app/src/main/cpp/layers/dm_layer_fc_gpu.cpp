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
                biases_multiplier_blob = new DM_Blob(vector<uint32_t>({(uint32_t)n}), ENVIRONMENT_GPU,
                                                     this->precision, biases_multiplier);
            }

            if(precision == PRECISION_32) {
                status = CLBlastSgemm(CLBlastLayoutRowMajor,
                                      CLBlastTransposeNo, CLBlastTransposeNo,
                                      batches, n, 1,
                                      1.0f,
                                      biases_multiplier_blob->get_gpu_data(), 0, 1,
                                      biases->get_gpu_data(), 0, n,
                                      1.0,
                                      data_out, 0, n,
                                      &queue, &event);
            } else if(precision == PRECISION_16) {
                status = CLBlastHgemm(CLBlastLayoutRowMajor,
                                      CLBlastTransposeNo, CLBlastTransposeNo,
                                      batches, n, 1,
                                      1.0f,
                                      biases_multiplier_blob->get_gpu_data(), 0, 1,
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
    }
}