#ifndef DM_LAYER_RELU_HPP
#define DM_LAYER_RELU_HPP

#include <dm_layer_param.hpp>
#include <dm_layer.hpp>
#include <dm_common.hpp>
#include <string>

namespace deepmon {
    class DM_Layer_ReLU : public DM_Layer {
    private:
    public:
        DM_Layer_ReLU(DM_Layer_Param &param);
        void LoadWeights() {}
        void ComputeOutputShapes(vector<vector<uint32_t >> inputs_shapes_no_batches);
        void PrintInfo() {
            LOGD("Layer: %s", this->name.c_str());
            LOGD("\tType: %s", this->type.c_str());
            LOGD("\tEnvironemt: %s", (env == ENVIRONMENT_CPU) ? "CPU" : "GPU");
            LOGD("\tPrecision: %d", (precision == PRECISION_32) ? 32 : 16);

            string inputs_str;
            for(int i = 0 ; i < this->bottom_layers.size() ; i++)
                inputs_str += this->bottom_layers.at(i) + " ";
            LOGD("\tInputs: [ %s ]", inputs_str.c_str());
        }
        DM_Blob *ForwardCpu(vector<DM_Blob *> blobs);
        DM_Blob *ForwardGpu(vector<DM_Blob *> blobs);
    };
}

#endif
