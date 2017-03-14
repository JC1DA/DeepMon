#ifndef DM_EXECUTION_ENGINE_CPU_HPP
#define DM_EXECUTION_ENGINE_CPU_HPP

#include "dm_execution_engine.hpp"
#include <string>

namespace deepmon {
    class DM_Execution_Engine_CPU : public DM_Execution_Engine {
        void im2col(const float* data_im, const uint32_t channels,
                        const uint32_t height, const uint32_t width, const uint32_t kernel_h,
                        const uint32_t kernel_w, const uint32_t pad_h, const uint32_t pad_w,
                        const uint32_t stride_h, const uint32_t stride_w,
                        const uint32_t dilation_h, const uint32_t dilation_w,
                    float* data_col);
    public:
        DM_Execution_Engine_CPU();
        void create_memory(DM_Blob *blob, float *initialized_data);
        void finalize_all_tasks();
        DM_Blob *blob_convert_to_cpu_blob(DM_Blob *blob);
        DM_Blob *blob_convert_to_gpu_blob(DM_Blob *blob, PRESICION_TYPE precision);
        void do_im2col(ENVIRONMENT_TYPE evn_type, MEMORY_LAYOUT mem_layout, DM_Blob *input, DM_Blob *output, \
            std::vector<uint32_t> filters_sizes, std::vector<uint32_t> strides, std::vector<uint32_t> pads, std::vector<uint32_t> dilations);
        void do_conv(MEMORY_LAYOUT mem_layout, DM_Blob *input, DM_Blob *output, \
            DM_Blob *filters, DM_Blob *biases, std::vector<uint32_t> strides, std::vector<uint32_t> pads, std::vector<uint32_t> dilations);
    };
}

#endif
