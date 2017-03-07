#ifndef DM_EXECUTION_ENGINE_CPU_HPP
#define DM_EXECUTION_ENGINE_CPU_HPP

#include "dm_execution_engine.hpp"
#include <string>

namespace deepmon {
    class DM_Execution_Engine_CPU : public DM_Execution_Engine {
    public:
        DM_Execution_Engine_CPU();
        void create_memory(DM_Blob *blob, float *initialized_data);
        void finalize_all_tasks();
        DM_Blob *blob_convert_to_cpu_blob(DM_Blob *blob);
        DM_Blob *blob_convert_to_gpu_blob(DM_Blob *blob, PRESICION_TYPE precision);
    };
}

#endif
