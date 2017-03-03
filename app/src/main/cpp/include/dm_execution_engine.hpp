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

class DM_Execution_Engine {
protected:
    ENVIRONMENT_TYPE type;
    bool initialized = false;
public:
    DM_Execution_Engine(ENVIRONMENT_TYPE type);
    //virtual void do_conv(void *input, void *params, void *output);
    //virtual void do_pooling(void *input, void *params, void *output);
    //virtual void do_fully_connected(void *input, void *params, void *output);
    //virtual void do_activation(void *input, void *params, void *output);
};

#endif /* DM_EXECUTION_ENGINE_HPP */

