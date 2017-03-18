//
// Created by JC1DA on 3/6/17.
//
#include <dm_blob.hpp>
#include <dm_common.hpp>
#include <dm.hpp>

namespace deepmon {
    DM_Blob::DM_Blob(std::vector<uint32_t> shapes, ENVIRONMENT_TYPE evn,
                     PRESICION_TYPE precision_type, float *initialized_data) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
        this->cpu_data = NULL;
        this->gpu_data = NULL;
        this->size = 1;
        for(std::vector<uint32_t >::iterator it = shapes.begin() ; it != shapes.end() ; it++) {
            this->shapes.push_back(*it);
            this->size *= *it;
        }
        this->environment = evn;
        this->precision = precision_type;

        deepmon::DeepMon::Get().create_memory(evn, this, initialized_data);
    }

    DM_Blob::~DM_Blob() {
        if(environment == ENVIRONMENT_CPU) {
            delete this->cpu_data;
            this->cpu_data = NULL;
        }

        if(environment == ENVIRONMENT_GPU) {
            clReleaseMemObject(this->gpu_data);
            this->gpu_data = NULL;
        }
    }
}

