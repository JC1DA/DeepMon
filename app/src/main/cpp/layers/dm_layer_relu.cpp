//
// Created by JC1DA on 4/11/17.
//

#include <layers/dm_layer_relu.hpp>
#include <json/json.h>
#include <fstream>
#include <dm.hpp>

using namespace std;
using namespace deepmon;

/*
 * Fixme: Just realize it's annoying to put USE_GPU flag inside configuration file
 */

namespace deepmon {
    DM_Layer_ReLU::DM_Layer_ReLU(DM_Layer_Param &param) : DM_Layer(param.GetName(), param.GetType(), param.GetInputLayersNames(), param.GetMemoryLayout()) {
        if(!param.GetConfPath().compare("")) {
            //no config files - use CPU implementation
            this->env = ENVIRONMENT_CPU;
            return;
        }

        ifstream in(param.GetConfPath().c_str());
        Json::Value layer;
        in >> layer;

        this->env = (layer["USE_GPU"].asBool()) ? ENVIRONMENT_GPU : ENVIRONMENT_CPU;
        if(this->env == ENVIRONMENT_GPU)
            this->precision = (layer["USE_HALF"].asBool()) ? PRECISION_16 : PRECISION_32;
    }

    void DM_Layer_ReLU::ComputeOutputShapes(vector<vector<uint32_t >> inputs_shapes_no_batches) {
        if(inputs_shapes_no_batches.size() != 1) {
            LOGE("Invalid Input's Shapes");
            this->corrupted = true;
            return;
        }

        vector<uint32_t> input_shapes = inputs_shapes_no_batches.at(0);

        //store this shape for loading weights
        this->inputs_shapes.push_back(input_shapes);
        this->output_shapes = input_shapes;
    }

    DM_Blob* DM_Layer_ReLU::ForwardCpu(vector<DM_Blob *> blobs) {
        if(blobs.size() != 1) {
            LOGE("[%s] has more than 1 input", this->name.c_str());
            return NULL;
        }

        /*
         * FIXME: For performance, we could re-use the input,
         * but we are not so sure if the input is used somewhere else or not (e.g. in resnet)
         */

        DM_Blob *input = blobs[0];
        DM_Blob *output = new DM_Blob(input->get_shapes(), ENVIRONMENT_CPU, PRECISION_32, NULL);

        float *data_in = input->get_cpu_data();
        float *data_out = output->get_cpu_data();
        for(int i = 0 ; i < output->get_total_size() ; i++)
            data_out[i] = data_in[i] > 0 ? data_in[i] : 0;

        return output;
    }

    DM_Blob* DM_Layer_ReLU::ForwardGpu(vector<DM_Blob *> blobs) {
        if(blobs.size() != 1) {
            LOGE("[%s] has more than 1 input", this->name.c_str());
            return NULL;
        }

        /*
         * FIXME: For performance, we could re-use the input,
         * but we are not so sure if the input is used somewhere else or not (e.g. in resnet)
         */

        DM_Blob *input = blobs[0];
        DM_Blob *output = new DM_Blob(input->get_shapes(), ENVIRONMENT_GPU, this->precision, NULL);

        DeepMon::Get().GetGpuExecutionEngine().ExecuteActivationReLU(this->mem_layout, this->precision, input, output);

        return output;
    }
}