/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   dm.hpp
 * Author: JC1DA
 *
 * Created on March 1, 2017, 4:49 PM
 */

#ifndef DM_HPP
#define DM_HPP

#include <string>
#include "dm_execution_engine_cpu.hpp"
#include "dm_execution_engine_gpu.hpp"
#include "dm_execution_engine.hpp"
#include "dm_blob.hpp"

using namespace std;

namespace deepmon {

    class DeepMon {
    private:
        DM_Execution_Engine *cpu_execution_engine = NULL;
        DM_Execution_Engine *gpu_execution_engine = NULL;

    public:
        DeepMon() {
            this->cpu_execution_engine = new DM_Execution_Engine_CPU();
            this->gpu_execution_engine = new DM_Execution_Engine_GPU();
        }

        DeepMon(std::string package_path) {
            this->cpu_execution_engine = new DM_Execution_Engine_CPU();
            this->gpu_execution_engine = new DM_Execution_Engine_GPU(package_path);
        }

        void create_memory(ENVIRONMENT_TYPE env_type, DM_Blob *blob, float *initialized_data);
        static DeepMon &Get();
        static DeepMon &Get(string package_path);
        DM_Execution_Engine &get_execution_engine(bool is_getting_cpu_engine) {
            if(is_getting_cpu_engine)
                return *cpu_execution_engine;
            else
                return *gpu_execution_engine;
        }
    };
}

#endif /* DM_HPP */

