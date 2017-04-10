//
// Created by JC1DA on 3/6/17.
//

#include <dm_execution_engine_cpu.hpp>
#include <dm_blob.hpp>

namespace deepmon {
    DM_Execution_Engine_CPU::DM_Execution_Engine_CPU() : DM_Execution_Engine(ENVIRONMENT_CPU) {
        this->initialized = true;
    }

    void DM_Execution_Engine_CPU::AllocateMemory(DM_Blob *blob, float *initialized_data) {
        if(blob->get_env() == this->evn) {
            uint32_t size_in_bytes = blob->get_size() * sizeof(float);
            blob->set_mem_size(size_in_bytes);

            float *data = new float[size_in_bytes / sizeof(float)];
            if(initialized_data != NULL)
                memcpy(data, initialized_data, size_in_bytes);
            blob->set_cpu_data(data);
        } else {
            blob->set_corrupted(true);
        }
    }
}