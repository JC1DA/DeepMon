#ifndef DM_LAYER_CONV_HPP
#define DM_LAYER_CONV_HPP

#include <dm_layer_param.hpp>
#include <dm_layer.hpp>
#include <dm_common.hpp>
#include <string>

using namespace std;

namespace deepmon {
    class DM_Layer_Conv : public DM_Layer {
    private:
        uint32_t num_filters;
        uint32_t num_channels;
        uint32_t filter_w;
        uint32_t filter_h;

        uint32_t pad_left = 0, pad_right = 0, pad_top = 0, pad_bottom = 0;
        uint32_t stride_w = 0, stride_h = 0;
        uint32_t dilation_h = 0, dilation_w = 0;

        bool has_bias = false;
        vector<uint32_t> filters_shapes;
        string weights_path;
        DM_Blob *filters;
        DM_Blob *biases;

        uint32_t input_h = 0;
        uint32_t input_w = 0;

        uint32_t output_h = 0;
        uint32_t output_w = 0;

        DM_Blob *do_conv_cpu(DM_Blob *input);
        DM_Blob *do_conv_gpu(DM_Blob *input);
        void CAFFE_LAYOUT_conv_gpu(DM_Blob *input, DM_Blob *output);
        void CAFFE_LAYOUT_im2col_cpu(DM_Blob *input, DM_Blob *output);
        void CAFFE_LAYOUT_im2col_gpu(DM_Blob *input, DM_Blob *output);
        void DM_LAYOUT_conv_gpu(DM_Blob *input, DM_Blob *output);
        void DM_LAYOUT_im2col_cpu(DM_Blob *input, DM_Blob *output);
        void DM_LAYOUT_conv_caching_gpu(DM_Blob *input, DM_Blob *output);
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
            LOGD("\tPads: [%d %d %d %d]", pad_left, pad_top, pad_right, pad_bottom);
            LOGD("\tStride: [%d %d]", stride_h, stride_w);
            LOGD("\tDilation: [%d %d]", dilation_h, dilation_w);

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
