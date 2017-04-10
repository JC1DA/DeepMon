#ifndef DM_LAYER_SOFTMAX_HPP
#define DM_LAYER_SOFTMAX_HPP

#include <dm_layer.hpp>
#include <dm_layer_param.hpp>

namespace deepmon {
    class DM_Layer_Softmax : public DM_Layer {
    private:
    protected:
    public:
        DM_Layer_Softmax(DM_Layer_Param &param);
        void ComputeOutputShapes(vector<vector<uint32_t >> inputs_shapes_no_batches);
        void LoadWeights() {}
        void PrintInfo() {
            LOGD("Layer: %s", this->name.c_str());
            LOGD("\tType: %s", this->type.c_str());
            LOGD("\tEnvironemt: CPU");
            LOGD("\tPrecision: 32");

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
