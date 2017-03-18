//
// Created by JC1DA on 3/18/17.
//

#include <json/json.h>
#include <fstream>
#include "layers/dm_layer_pooling.hpp"

namespace deepmon {
    DM_Layer_Pooling::DM_Layer_Pooling(DM_Layer_Param &param) : DM_Layer(param.GetName(), param.GetType(), param.GetInputLayersNames(), param.GetMemoryLayout()) {
        //read config file
        ifstream in(param.GetConfPath().c_str());
        Json::Value layer;
        in >> layer;

        this->type = layer["TYPE"].asString();

        this->filter_h = layer["FILTER_H"].asUInt();
        this->filter_w = layer["FILTER_W"].asUInt();

        this->env = (layer["USE_GPU"].asBool()) ? ENVIRONMENT_GPU : ENVIRONMENT_CPU;
        if(this->env == ENVIRONMENT_GPU)
            this->precision = (layer["USE_HALF"].asBool()) ? PRECISION_16 : PRECISION_32;

        for (Json::Value::iterator it = layer["FILTER_PADS"].begin(); it != layer["FILTER_PADS"].end(); ++it) {
            this->pads.push_back((*it).asUInt());
        }
        for (Json::Value::iterator it = layer["FILTER_STRIDES"].begin(); it != layer["FILTER_STRIDES"].end(); ++it) {
            this->strides.push_back((*it).asUInt());
        }
        for (Json::Value::iterator it = layer["FILTER_DILATIONS"].begin(); it != layer["FILTER_DILATIONS"].end(); ++it) {
            this->dilations.push_back((*it).asUInt());
        }

        if(dilations.size() == 0) {
            dilations.push_back(1);
            dilations.push_back(1);
        }

        if(pads.size() == 2) {
            pads.push_back(pads.at(0));
            pads.push_back(pads.at(1));
        }

        if(pads.size() != 4 || strides.size() != 2 || dilations.size() != 2) {
            corrupted = true;
            return;
        }

        bool found_correct_type = false;
        for(int i = 0 ; i < all_types.size() ; i++) {
            if(!all_types[i].compare(this->type)) {
                found_correct_type = true;
                switch(i) {
                    case 0:
                        this->Forward_Pooling = &Forward_MaxPool;
                        break;
                    case 1:
                        this->Forward_Pooling = &Forward_AvgPool;
                        break;
                }
            }
        }
        if(!found_correct_type) {
            this->corrupted = true;
            LOGE("Incorrect type of pooling layer");
        }
    }

    void DM_Layer_Pooling::ComputeOutputShapes(vector<vector<uint32_t >> inputs_shapes_no_batches) {
        if(inputs_shapes_no_batches.size() != 1) {
            LOGE("Invalid Input's Shapes");
            this->corrupted = true;
            return;
        }

        vector<uint32_t> input_shapes = inputs_shapes_no_batches.at(0);
        uint32_t input_channels;
        uint32_t input_h;
        uint32_t input_w;
        if(mem_layout == MEMORY_LAYOUT_CAFFE) {
            input_channels = input_shapes.at(0);
            input_h = input_shapes.at(1);
            input_w = input_shapes.at(2);
        }
        else if(mem_layout == MEMORY_LAYOUT_DM) {
            input_channels = input_shapes.at(2);
            input_h = input_shapes.at(0);
            input_w = input_shapes.at(1);
        }

        const int output_h = (input_h + pads[1] + pads[3] - (dilations[0] * (filter_h - 1) + 1)) / strides[0] + 1;
        const int output_w = (input_w + pads[0] + pads[2] - (dilations[1] * (filter_w - 1) + 1)) / strides[1] + 1;

        if(output_h <= 0 || output_w <= 0) {
            LOGE("%s: Incorrect output size (%d, %d)", output_h, output_w);
            this->corrupted = true;
            return;
        }

        if(mem_layout == MEMORY_LAYOUT_DM) {
            this->output_shapes.push_back(output_h);
            this->output_shapes.push_back(output_w);
            this->output_shapes.push_back(input_channels);
        } else if(mem_layout == MEMORY_LAYOUT_CAFFE){
            this->output_shapes.push_back(input_channels);
            this->output_shapes.push_back(output_h);
            this->output_shapes.push_back(output_w);
        }
    }

    void DM_Layer_Pooling::Forward_MaxPool(DM_Blob *input, DM_Blob *output) {

    }

    void DM_Layer_Pooling::Forward_AvgPool(DM_Blob *input, DM_Blob *output) {
        
    }
}