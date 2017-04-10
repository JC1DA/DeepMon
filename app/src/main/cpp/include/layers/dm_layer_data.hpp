#ifndef DM_LAYER_DATA_HPP
#define DM_LAYER_DATA_HPP

#include <dm_layer.hpp>
#include "dm_layer_param.hpp"

namespace deepmon {
    class DM_Layer_Data : public DM_Layer {
    private:
        uint32_t input_w = -1, input_h = -1, input_c = -1;
    protected:
    public:
        DM_Layer_Data(DM_Layer_Param &param);
        void ComputeOutputShapes(vector<vector<uint32_t >> inputs_shapes_no_batches);
        void LoadWeights() {

        }
        void PrintInfo() {
            LOGD("Layer: %s", this->name.c_str());
            LOGD("\tType: %s", this->type.c_str());
            LOGD("\tEnvironemt: %s", (env == ENVIRONMENT_CPU) ? "CPU" : "GPU");
            LOGD("\tPrecision: %d", (precision == PRECISION_32) ? 32 : 16);
            LOGD("\tInput: [%d %d %d]", input_c, input_h, input_w);
        }
        DM_Blob *ForwardCpu(vector<DM_Blob *> blobs);
        DM_Blob *ForwardGpu(vector<DM_Blob *> blobs);
    };
}

#endif