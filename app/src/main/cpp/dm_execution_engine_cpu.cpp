//
// Created by JC1DA on 3/6/17.
//

#include <dm_execution_engine_cpu.hpp>
#include <dm_blob.hpp>

namespace deepmon {
    DM_Execution_Engine_CPU::DM_Execution_Engine_CPU() : DM_Execution_Engine(ENVIRONMENT_CPU) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
    }

    void DM_Execution_Engine_CPU::create_memory(DM_Blob *blob, float *initialized_data) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
        if(!blob->is_corrupted() && blob->get_env() == this->type) {
            int size_in_bytes = blob->get_size() * sizeof(float);
            blob->set_mem_size(size_in_bytes);

            float *data = new float[size_in_bytes / sizeof(float)];
            if(initialized_data != NULL)
                memcpy(data, initialized_data, size_in_bytes);
            blob->set_cpu_data(data);
        } else {
            blob->set_corrupted(true);
        }
    }

    void DM_Execution_Engine_CPU::finalize_all_tasks() {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
    }

    DM_Blob *DM_Execution_Engine_CPU::blob_convert_to_cpu_blob(DM_Blob *blob) {
        return NULL;
    }

    DM_Blob *DM_Execution_Engine_CPU::blob_convert_to_gpu_blob(DM_Blob *blob, PRESICION_TYPE precision){
        return NULL;
    }
}