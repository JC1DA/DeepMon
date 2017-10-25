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

#include "layers/dm_layer_data.hpp"
#include <fstream>
#include <json/json.h>

using namespace deepmon;
namespace deepmon {
    DM_Layer_Data::DM_Layer_Data(DM_Layer_Param &param) : DM_Layer(param.GetName(), param.GetType(), param.GetInputLayersNames(), param.GetMemoryLayout()) {
        //read config file
        ifstream in(param.GetConfPath().c_str());
        Json::Value layer;
        in >> layer;

        this->input_w = layer["INPUT_W"].asUInt();
        this->input_h = layer["INPUT_H"].asUInt();
        this->input_c = layer["INPUT_C"].asUInt();

        this->env = (layer["USE_GPU"].asBool()) ? ENVIRONMENT_GPU : ENVIRONMENT_CPU;
        if(this->env == ENVIRONMENT_GPU)
            this->precision = (layer["USE_HALF"].asBool()) ? PRECISION_16 : PRECISION_32;
    }

    void DM_Layer_Data::ComputeOutputShapes(vector<vector<uint32_t >> inputs_shapes_no_batches) {
        //ignore the input for this layer
        if(mem_layout == MEMORY_LAYOUT_DM) {
            this->output_shapes.push_back(this->input_h);
            this->output_shapes.push_back(this->input_w);
            this->output_shapes.push_back(this->input_c);
        } else if(mem_layout == MEMORY_LAYOUT_CAFFE) {
            this->output_shapes.push_back(this->input_c);
            this->output_shapes.push_back(this->input_h);
            this->output_shapes.push_back(this->input_w);
        }
    }

    DM_Blob* DM_Layer_Data::ForwardCpu(vector<DM_Blob *> blobs) {
        if(blobs.size() != 1) {
            LOGE("[%s]: has more than 1 input", name.c_str());
            return NULL;
        }

        DM_Blob *input = blobs[0];

        if(input == NULL || input->get_shapes().size() != 4) {
            LOGE("[%s]: Invalid Input Shape !!!", name.c_str());
            return NULL;
        }

        if(input->get_shape_at(1) != output_shapes[0] || input->get_shape_at(2) != output_shapes[1] || input->get_shape_at(3) != output_shapes[2]) {
            LOGE("[%s]: Invalid Input Shape !!!", name.c_str());
            return NULL;
        }

        /*
         * For better performance, we forward this blob to next layer
         * To prevent this blob being deleted, we need to set it to persistent blob even though this layer does not require
         */
        input->set_persistent(true);

        return input;
    }

    DM_Blob* DM_Layer_Data::ForwardGpu(vector<DM_Blob *> blobs) {
        if(blobs.size() != 1) {
            LOGE("[%s]: has more than 1 input", name.c_str());
            return NULL;
        }

        DM_Blob *input = blobs[0];

        if(input == NULL || input->get_shapes().size() != 4) {
            LOGE("[%s]: Invalid Input Shape !!!", name.c_str());
            return NULL;
        }

        if(input->get_shape_at(1) != output_shapes[0] || input->get_shape_at(2) != output_shapes[1] || input->get_shape_at(3) != output_shapes[2]) {
            LOGE("[%s]: Invalid Input Shape !!!", name.c_str());
            return NULL;
        }

        input->set_persistent(true);

        return input;
    }

    DM_Blob* DM_Layer_Data::ForwardCache(vector<DM_Blob *> blobs) {
        return ForwardGpu(blobs);
    }
}