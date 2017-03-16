#ifndef DM_LAYER_CONV_HPP
#define DM_LAYER_CONV_HPP

#include "dm_layer.hpp"

namespace deepmon {
    class DM_Layer_Conv : public DM_Layer {
    private:
        uint32_t num_filters;
        uint32_t num_channels;
        uint32_t filter_w;
        uint32_t filter_h;

        vector<uint32_t> pads;
        vector<uint32_t> strides;
        vector<uint32_t> dilations;

        bool has_bias = false;
        vector<uint32_t> filters_shapes;
        string weights_path;
        DM_Blob *filters;
        DM_Blob *biases;
    protected:
    public:
        DM_Layer_Conv(DM_Layer_Param &param);
        void LoadWeights();
        void ComputeOutputShapes(vector<vector<uint32_t >> inputs_shapes_no_batches);
        void PrintInfo() {
            LOGD("Layer: %s", this->name.c_str());
            LOGD("\tType: %s", this->type.c_str());
            LOGD("\tEnvironemt: %s", (env == ENVIRONMENT_CPU) ? "CPU" : "GPU");
            LOGD("\tPrecision: %d", (precision == PRECISION_32) ? 32 : 16);
            LOGD("\tFilter's dims: [%d %d %d %d]", num_filters, num_channels, filter_h, filter_w);
            LOGD("\tPads: [%d %d %d %d]", pads.at(0), pads.at(1), pads.at(2), pads.at(3));
            LOGD("\tStride: [%d %d]", strides.at(0), strides.at(1));
            LOGD("\tDilation: [%d %d]", dilations.at(0), dilations.at(1));

            string inputs_str;
            for(int i = 0 ; i < this->bottom_layers.size() ; i++)
                inputs_str += this->bottom_layers.at(i) + " ";
            LOGD("\tInputs: [ %s ]", inputs_str.c_str());
        }
    };
}

#endif
