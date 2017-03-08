#ifndef DM_EXECUTION_ENGINE_CPU_HPP
#define DM_EXECUTION_ENGINE_CPU_HPP

#include "dm_execution_engine.hpp"
#include <string>

namespace deepmon {
    class DM_Execution_Engine_CPU : public DM_Execution_Engine {
        void im2col(const float* data_im, const int channels,
                        const int height, const int width, const int kernel_h,
                        const int kernel_w, const int pad_h, const int pad_w,
                        const int stride_h, const int stride_w,
                        const int dilation_h, const int dilation_w,
                    float* data_col);
    public:
        DM_Execution_Engine_CPU();
        void create_memory(DM_Blob *blob, float *initialized_data);
        void finalize_all_tasks();
        DM_Blob *blob_convert_to_cpu_blob(DM_Blob *blob);
        DM_Blob *blob_convert_to_gpu_blob(DM_Blob *blob, PRESICION_TYPE precision);
        void do_conv(DM_Blob *input, DM_Blob *output, DM_Blob *filters, DM_Blob *biases, std::vector<int> strides, std::vector<int> pads, std::vector<int> dilations);
    };
}

#endif
