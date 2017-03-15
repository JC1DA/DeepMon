//
// Created by JC1DA on 3/15/17.
//
#include "layers/dm_layer_data.hpp"
#include <fstream>
#include <json/json.h>

using namespace deepmon;
namespace deepmon {
    DM_Layer_Data::DM_Layer_Data(DM_Layer_Param &param) : DM_Layer(param.GetName(), param.GetType(), param.GetInputLayersNames()) {
        this->param = &param;
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
}