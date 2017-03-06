//
// Created by JC1DA on 3/6/17.
//

#include <dm_blob.hpp>
#include <vector>
#include <dm_log.hpp>
#include <string.h>
#include "CL/cl.h"

DM_Blob::DM_Blob(std::vector<int> shapes, ENVIRONMENT_TYPE evn, PRESICION_TYPE precision_type, char * initialized_data) {
    this->size = 1;
    for(std::vector<int>::iterator it = shapes.begin() ; it != shapes.end() ; it++) {
        this->shapes.push_back(*it);
        this->size *= *it;
    }
    this->environment = evn;
    this->precision = precision_type;

    if(this->environment == ENVIRONMENT_CPU) {
        create_Cpu_mem(initialized_data);
    } else if(this->environment == ENVIRONMENT_GPU) {
        create_Gpu_mem(this->precision, initialized_data);
    } else {
        LOGE("Unrecognized Environment !!!");
        this->is_corrupted = true;
    }
}

void DM_Blob::create_Cpu_mem(char * initialized_data) {
    this->mem_size = this->size * sizeof(float);
    this->cpu_data = new char[this->mem_size];
    if(initialized_data != NULL)
        memcpy(this->cpu_data, initialized_data, this->mem_size);
}

void DM_Blob::create_Gpu_mem(PRESICION_TYPE type, char * initialized_data) {
    if(type != PRECISION_16 && type != PRECISION_32) {
        this->is_corrupted = true;
        return;
    }

    if(type == PRECISION_32)
        this->mem_size = this->size * sizeof(cl_float);
    else if(type == PRECISION_16)
        this->mem_size = this->size * sizeof(cl_half);


}