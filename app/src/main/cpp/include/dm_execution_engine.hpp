/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   dm_execution_engine.hpp
 * Author: JC1DA
 *
 * Created on March 1, 2017, 8:07 PM
 */

#ifndef DM_EXECUTION_ENGINE_HPP
#define DM_EXECUTION_ENGINE_HPP

#include "dm_common.hpp"
#include "dm_blob.hpp"

namespace deepmon {
    class DM_Execution_Engine {
    protected:
        ENVIRONMENT_TYPE type;
        bool initialized = false;
    public:
        DM_Execution_Engine(ENVIRONMENT_TYPE type);
        virtual void create_memory(DM_Blob *blob, float *initialized_data)=0;
        virtual DM_Blob *blob_convert_to_cpu_blob(DM_Blob *blob) = 0;
        virtual DM_Blob *blob_convert_to_gpu_blob(DM_Blob *blob, PRESICION_TYPE precision) = 0;
        virtual void finalize_all_tasks() = 0;
        virtual void do_conv(DM_Blob *input, DM_Blob *output, DM_Blob *filters, DM_Blob *biases, std::vector<int> strides, std::vector<int> pads, std::vector<int> dilations) = 0;
        //virtual void do_pooling(void *input, void *params, void *output);
        //virtual void do_fully_connected(void *input, void *params, void *output);
        //virtual void do_activation(void *input, void *params, void *output);
    };
}

#endif /* DM_EXECUTION_ENGINE_HPP */

