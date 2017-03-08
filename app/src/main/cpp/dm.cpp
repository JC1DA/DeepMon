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
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
        this->cpu_execution_engine = new DM_Execution_Engine_CPU();
        this->gpu_execution_engine = new DM_Execution_Engine_GPU();
    }

    DeepMon::DeepMon(std::string package_path) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
        this->cpu_execution_engine = new DM_Execution_Engine_CPU();
        this->gpu_execution_engine = new DM_Execution_Engine_GPU(package_path);
    }

    DM_Execution_Engine* DeepMon::get_excution_engine(bool is_cpu_engine) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
        if(is_cpu_engine)
            return this->cpu_execution_engine;
        else
            return this->gpu_execution_engine;
    }

    DeepMon &DeepMon::Get() {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
        if(dm == NULL) {
            dm = new DeepMon();
        }

        return *dm;
    }

    DeepMon &DeepMon::Get(std::string package_path) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
        if(dm == NULL) {
            dm = new DeepMon(package_path);
        }

        return *dm;
    }
}



