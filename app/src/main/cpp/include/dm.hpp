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

#include "dm_execution_engine.hpp"
#include <string>

namespace deepmon {

    class DeepMon {
    private:
        DM_Execution_Engine *cpu_execution_engine;
        DM_Execution_Engine *gpu_execution_engine;
    public:
        DeepMon();
        DeepMon(std::string package_path);
        DM_Execution_Engine *get_excution_engine(bool is_cpu_engine);
        static DeepMon &Get();
        static DeepMon &Get(std::string package_path);
    };
}

#endif /* DM_HPP */

