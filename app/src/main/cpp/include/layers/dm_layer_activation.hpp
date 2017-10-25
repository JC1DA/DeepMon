#ifndef DM_LAYER_ACTIVATION_HPP
#define DM_LAYER_ACTIVATION_HPP

#include <dm_layer.hpp>
#include <dm_layer_param.hpp>

using namespace std;

#define ACTIVATION_RELU_STR                 "RELU"
#define ACTIVATION_LEAKY_STR                "LEAKY"

#define ACTIVATION_RELU                     0
#define ACTIVATION_LEAKY                    1

namespace deepmon {
    class DM_Layer_Activation : public DM_Layer {
    private:
        int activation_type;
        float activation_threshold; //used for leaky activation
        void Activation_ReLU_CPU(DM_Blob *input, DM_Blob *output);
        void Activation_ReLU_GPU(DM_Blob *input, DM_Blob *output);
        void Activation_Leaky_CPU(DM_Blob *input, DM_Blob *output);
        void Activation_Leaky_GPU(DM_Blob *input, DM_Blob *output);
    public:
        DM_Layer_Activation(DM_Layer_Param &param);
        void LoadWeights() {}
        void ComputeOutputShapes(vector<vector<uint32_t >> inputs_shapes_no_batches);
        void PrintInfo() {
            LOGD("Layer: %s", this->name.c_str());
            LOGD("\tType: %s", this->type.c_str());
            LOGD("\tActivation Type: ");
            LOGD("\tEnvironemt: %s", (env == ENVIRONMENT_CPU) ? "CPU" : "GPU");
            LOGD("\tPrecision: %d", (precision == PRECISION_32) ? 32 : 16);

            string inputs_str;
            for(int i = 0 ; i < this->bottom_layers.size() ; i++)
                inputs_str += this->bottom_layers.at(i) + " ";
            LOGD("\tInputs: [ %s ]", inputs_str.c_str());
        }
        DM_Blob *ForwardCpu(vector<DM_Blob *> blobs);
        DM_Blob *ForwardGpu(vector<DM_Blob *> blobs);
        DM_Blob *ForwardCache(vector<DM_Blob *> blobs);
    };
}

#endif
