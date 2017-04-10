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
        ENVIRONMENT_TYPE evn;
        bool initialized = false;
    public:
        DM_Execution_Engine(ENVIRONMENT_TYPE evn) {
            this->evn = evn;
        }
        bool IsWorking() {
            return this->initialized;
        }
        virtual void AllocateMemory(DM_Blob *blob, float *initialized_data) = 0;
    };
}

#endif /* DM_EXECUTION_ENGINE_HPP */

