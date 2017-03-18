/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <dm_execution_engine_cpu.hpp>
#include <dm_execution_engine_gpu.hpp>
#include "dm.hpp"

namespace deepmon {
    static DeepMon *dm;

    void DeepMon::create_memory(ENVIRONMENT_TYPE env_type, DM_Blob *blob,
                                float *initialized_data) {
        if(env_type == ENVIRONMENT_CPU)
            this->cpu_execution_engine->create_memory(blob, initialized_data);
        else if(env_type == ENVIRONMENT_GPU)
            this->gpu_execution_engine->create_memory(blob, initialized_data);
        else
            blob->set_corrupted(true);
    }

    DeepMon &DeepMon::Get() {
        if(dm == NULL) {
            dm = new DeepMon();
        }

        return *dm;
    }

    DeepMon &DeepMon::Get(std::string package_path) {
        if(dm == NULL) {
            dm = new DeepMon(package_path);
        }

        return *dm;
    }
}



