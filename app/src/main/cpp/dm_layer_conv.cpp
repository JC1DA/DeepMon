//
// Created by JC1DA on 3/13/17.
//

#include <layers/dm_layer_conv.hpp>
#include <json/json.h>
#include <fstream>

namespace deepmon {
    DM_Layer_Conv::DM_Layer_Conv(DM_Layer_Param &param) : DM_Layer() {
        //read config file
        ifstream in(param.GetConfPath().c_str());
        Json::Value layer;
        in >> layer;

        this->num_filters = layer["NUM_FILTERS"].asUInt();
        this->num_channels = layer["NUM_CHANNELS"].asUInt();
        this->filter_h = layer["FILTER_H"].asUInt();
        this->filter_w = layer["FILTER_W"].asUInt();

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

        //read weights
    }

    void DM_Layer_Conv::Forward_CPU(const std::vector<DM_Blob *> &bottom,
                                    const std::vector<DM_Blob *> &top) {

    }

    void DM_Layer_Conv::Forward_GPU(const std::vector<DM_Blob *> &bottom,
                                    const std::vector<DM_Blob *> &top) {

    }

    void DM_Layer_Conv::LayerSetUp(const std::vector<DM_Blob *> &bottom,
                                   const std::vector<DM_Blob *> &top) {

    }

    void DM_Layer_Conv::Reshape(const std::vector<DM_Blob *> &bottom,
                                const std::vector<DM_Blob *> &top) {

    }
}