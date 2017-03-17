//
// Created by JC1DA on 3/13/17.
//

#include <layers/dm_layer_conv.hpp>
#include <json/json.h>
#include <fstream>

using namespace deepmon;

namespace deepmon {
    DM_Layer_Conv::DM_Layer_Conv(DM_Layer_Param &param) : DM_Layer(param.GetName(), param.GetType(), param.GetInputLayersNames(), param.GetMemoryLayout()) {
        //read config file
        ifstream in(param.GetConfPath().c_str());
        Json::Value layer;
        in >> layer;

        //save weights path
        this->weights_path = param.GetWeightsPath();

        this->num_filters = layer["NUM_FILTERS"].asUInt();
        this->num_channels = layer["NUM_CHANNELS"].asUInt();
        this->filter_h = layer["FILTER_H"].asUInt();
        this->filter_w = layer["FILTER_W"].asUInt();
        this->has_bias = layer["HAS_BIAS"].asBool();

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

        if(num_filters <= 0 || num_channels <= 0 || filter_h <= 0 || filter_w <= 0 ) {
            corrupted = true;
            return;
        }


        switch (param.GetMemoryLayout()) {
            case MEMORY_LAYOUT_DM:
                this->filters_shapes.push_back(num_filters);
                this->filters_shapes.push_back(filter_h);
                this->filters_shapes.push_back(filter_w);
                this->filters_shapes.push_back(num_channels);
                break;
            case MEMORY_LAYOUT_CAFFE:
                this->filters_shapes.push_back(num_filters);
                this->filters_shapes.push_back(num_channels);
                this->filters_shapes.push_back(filter_h);
                this->filters_shapes.push_back(filter_w);
                break;
            default:
                LOGE("Invalid Memory Layout");
        }
    }

    void DM_Layer_Conv::LoadWeights() {
        float *bias_data = NULL;
        float *weights_data = NULL;

        FILE *fp = fopen(this->weights_path.c_str(), "r");
        if(this->has_bias) {
            bias_data = new float[this->num_filters];
            fread((void*)bias_data, sizeof(float), this->num_filters, fp);
            this->biases = new DM_Blob(vector<uint32_t>(this->num_filters), this->env, this->precision, bias_data);
            delete bias_data;
        }
        weights_data = new float[this->num_filters * this->num_channels * this->filter_h * this->filter_w];
        fread((void*)weights_data, sizeof(float), this->num_filters * this->num_channels * this->filter_h * this->filter_w, fp);
        fclose(fp);

        this->filters = new DM_Blob(this->filters_shapes, this->env, this->precision, weights_data);
        delete weights_data;
    }

    void DM_Layer_Conv::ComputeOutputShapes(vector<vector<uint32_t >> inputs_shapes_no_batches) {
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

        if(this->num_channels != input_channels) {
            LOGE("%s: Incorrect number of channels (%d != %d)", this->name.c_str(), this->num_channels, input_channels);
            this->corrupted = true;
            return;
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
            this->output_shapes.push_back(num_filters);
        } else if(mem_layout == MEMORY_LAYOUT_CAFFE){
            this->output_shapes.push_back(num_filters);
            this->output_shapes.push_back(output_h);
            this->output_shapes.push_back(output_w);
        }
    }
}