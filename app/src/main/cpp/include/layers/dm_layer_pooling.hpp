#ifndef DM_LAYER_POOLING_HPP
#define DM_LAYER_POOLING_HPP

#include <dm_layer.hpp>
#include <dm_layer_param.hpp>

namespace deepmon {
    class DM_Layer_Pooling : public DM_Layer {
    private:
        vector<string> all_types{
           string("MAXPOOL"),
           string("AVGPOOL")
        };
        uint32_t filter_w;
        uint32_t filter_h;

        uint32_t pad_left = 0, pad_right = 0, pad_top = 0, pad_bottom = 0;
        uint32_t stride_w = 0, stride_h = 0;

        //uint32_t dilation_h = 0, dilation_w = 0;

        uint32_t num_channels = 0;
        uint32_t input_h = 0;
        uint32_t input_w = 0;

        uint32_t output_h = 0;
        uint32_t output_w = 0;

        string type;
        void (DM_Layer_Pooling::*Forward_Pooling)(DM_Blob *input, DM_Blob *output);

        void CAFFE_LAYOUT_ForwardCPU_MaxPool(DM_Blob *input, DM_Blob *output);
        void CAFFE_LAYOUT_ForwardCPU_AvePool(DM_Blob *input, DM_Blob *output);
        void DM_LAYOUT_ForwardCPU_MaxPool(DM_Blob *input, DM_Blob *output);
        void DM_LAYOUT_ForwardCPU_AvePool(DM_Blob *input, DM_Blob *output);
        void CAFFE_LAYOUT_ForwardGPU(DM_Blob *input, DM_Blob *output);
    protected:
    public:
        DM_Layer_Pooling(DM_Layer_Param &param);
        void ComputeOutputShapes(vector<vector<uint32_t >> inputs_shapes_no_batches);
        void LoadWeights() {}
        void PrintInfo() {
            LOGD("Layer: %s", this->name.c_str());
            LOGD("\tType: %s", this->type.c_str());
            LOGD("\tEnvironemt: %s", (env == ENVIRONMENT_CPU) ? "CPU" : "GPU");
            LOGD("\tPrecision: %d", (precision == PRECISION_32) ? 32 : 16);
            LOGD("\tPads: [%d %d %d %d]", pad_left, pad_top, pad_right, pad_bottom);
            LOGD("\tStride: [%d %d]", stride_h, stride_w);

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


