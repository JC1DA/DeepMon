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
        DM_Blob *filters;
        DM_Blob *biases;
    protected:
        void Forward_CPU(
                const std::vector<DM_Blob *> &bottom,
                const std::vector<DM_Blob *> &top
        );
        void Forward_GPU(
                const std::vector<DM_Blob *> &bottom,
                const std::vector<DM_Blob *> &top
        ) ;
    public:
        DM_Layer_Conv(DM_Layer_Param &param);

        void LayerSetUp(
                const std::vector<DM_Blob*>& bottom,
                const std::vector<DM_Blob*>& top);

        void Reshape(
                const std::vector<DM_Blob*>& bottom,
                const std::vector<DM_Blob*>& top
        );
    };
}

#endif
