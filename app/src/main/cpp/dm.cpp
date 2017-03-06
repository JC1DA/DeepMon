/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "dm.hpp"
#include "dm_execution_engine_gpu.hpp"

DeepMon::DeepMon() {
    this->cpu_execution_engine = NULL;
    this->gpu_execution_engine = new DM_Execution_Engine_GPU();
}

DeepMon::DeepMon(std::string package_path) {
    //this->cpu_execution_engine = NULL;
    this->gpu_execution_engine = new DM_Execution_Engine_GPU(package_path);
}

