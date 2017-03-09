/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <dm_log.hpp>
#include "dm_configs.hpp"
#include "dm_execution_engine_cpu.hpp"
#include "dm.hpp"
#include "dm_execution_engine_gpu.hpp"
#include "dm_common.hpp"

namespace deepmon {
    static DeepMon *dm;

    DeepMon::DeepMon() {
        this->cpu_execution_engine = new DM_Execution_Engine_CPU();
        this->gpu_execution_engine = new DM_Execution_Engine_GPU();
    }

    DeepMon::DeepMon(std::string package_path) {
        this->cpu_execution_engine = new DM_Execution_Engine_CPU();
        this->gpu_execution_engine = new DM_Execution_Engine_GPU(package_path);
    }

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



