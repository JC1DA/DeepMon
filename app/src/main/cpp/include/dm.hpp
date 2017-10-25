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
#include "dm_blob.hpp"
#include "dm_execution_engine.hpp"
#include "dm_execution_engine_cpu.hpp"
#include "dm_execution_engine_gpu.hpp"

using namespace std;

namespace deepmon {

    class DeepMon {
    private:
        DM_Execution_Engine_CPU *cpu_execution_engine = NULL;
        DM_Execution_Engine_GPU *gpu_execution_engine = NULL;

    public:
        DeepMon() {
            this->cpu_execution_engine = new DM_Execution_Engine_CPU();
            this->gpu_execution_engine = new DM_Execution_Engine_GPU();
        }
        DeepMon(std::string package_path) {
            this->cpu_execution_engine = new DM_Execution_Engine_CPU();
            this->gpu_execution_engine = new DM_Execution_Engine_GPU(package_path);
        }

        static DeepMon &Get();
        static DeepMon &Get(string package_path);

        DM_Execution_Engine_CPU &GetCpuExecutionEngine() {
            return *cpu_execution_engine;
        }

        DM_Execution_Engine_GPU &GetGpuExecutionEngine() {
            return *gpu_execution_engine;
        }

        void AllocateMemory(ENVIRONMENT_TYPE env_type, DM_Blob *blob, float *initialized_data) {
            if(blob->get_env() == ENVIRONMENT_CPU) {
                this->cpu_execution_engine->AllocateMemory(blob, initialized_data);
            } else if(blob->get_env() == ENVIRONMENT_GPU) {
                this->gpu_execution_engine->AllocateMemory(blob, initialized_data);
            } else {
                LOGE("Unsupported Environment");
                blob->set_corrupted(true);
            }
        }
        DM_Blob *ConvertBlob(DM_Blob *blob, ENVIRONMENT_TYPE to_evn, PRESICION_TYPE to_precision) {
            DM_Blob *result = NULL;

            if(blob->get_env() == ENVIRONMENT_CPU && to_evn == ENVIRONMENT_CPU) {
                //clone a blob
                result = new DM_Blob(blob->get_shapes(), to_evn, to_precision, blob->get_cpu_data());
                return result;
            }

            if(!this->gpu_execution_engine->IsWorking())
                return NULL;

            if(to_evn == ENVIRONMENT_CPU)
                result = this->gpu_execution_engine->blob_convert_to_cpu_blob(blob);
            else if(to_evn == ENVIRONMENT_GPU)
                result = this->gpu_execution_engine->blob_convert_to_gpu_blob(blob, to_precision);

            return result;
        }
    };
}

#endif /* DM_HPP */

