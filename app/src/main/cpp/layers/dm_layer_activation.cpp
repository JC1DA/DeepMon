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
#include <json/json.h>
#include <fstream>

namespace deepmon {
    DM_Layer_Activation::DM_Layer_Activation(DM_Layer_Param &param) : DM_Layer(param.GetName(), param.GetType(), param.GetInputLayersNames(), param.GetMemoryLayout()) {
        if(!param.GetConfPath().compare("")) {
            //no config files - broken layer
            this->corrupted = true;
            return;
        }

        ifstream in(param.GetConfPath().c_str());
        Json::Value layer;
        in >> layer;

        if(!layer["type"].asString().compare(ACTIVATION_RELU_STR)) {
            this->activation_type = ACTIVATION_RELU;
            this->activation_threshold = 0;
        } else if(!layer["type"].asString().compare(ACTIVATION_LEAKY_STR)) {
            this->activation_type = ACTIVATION_LEAKY;
            this->activation_threshold = layer["threshold"].asFloat();
        } else {
            //unsupported activation
            this->corrupted = true;
            return;
        }

        this->env = (layer["USE_GPU"].asBool()) ? ENVIRONMENT_GPU : ENVIRONMENT_CPU;
        if(this->env == ENVIRONMENT_GPU)
            this->precision = (layer["USE_HALF"].asBool()) ? PRECISION_16 : PRECISION_32;

    }

    void DM_Layer_Activation::ComputeOutputShapes(vector<vector<uint32_t >> inputs_shapes_no_batches) {
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

    DM_Blob* DM_Layer_Activation::ForwardCpu(vector<DM_Blob *> blobs) {
        if(blobs.size() != 1) {
            LOGE("[%s] has more than 1 input", this->name.c_str());
            return NULL;
        }

        DM_Blob *input = blobs[0];
        DM_Blob *output = new DM_Blob(input->get_shapes(), ENVIRONMENT_CPU, PRECISION_32, NULL);

        switch(activation_type) {
            case ACTIVATION_RELU:
                Activation_ReLU_CPU(input, output);
                break;
            case ACTIVATION_LEAKY:
                Activation_Leaky_CPU(input, output);
                break;
            default:
                output->set_corrupted(true);
                break;
        }

        return output;
    }

    DM_Blob* DM_Layer_Activation::ForwardGpu(vector<DM_Blob *> blobs) {
        if(blobs.size() != 1) {
            LOGE("[%s] has more than 1 input", this->name.c_str());
            return NULL;
        }

        DM_Blob *input = blobs[0];
        DM_Blob *output = new DM_Blob(input->get_shapes(), ENVIRONMENT_GPU, this->precision, NULL);

        switch(activation_type) {
            case ACTIVATION_RELU:
                Activation_ReLU_GPU(input, output);
                break;
            case ACTIVATION_LEAKY:
                Activation_Leaky_GPU(input, output);
                break;
            default:
                output->set_corrupted(true);
                break;
        }

        return output;
    }

    DM_Blob* DM_Layer_Activation::ForwardCache(vector<DM_Blob *> blobs) {
        return ForwardGpu(blobs);
    }
}