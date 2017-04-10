//
// Created by JC1DA on 3/6/17.
//
#include <dm_blob.hpp>
#include <dm_common.hpp>
#include <dm_log.hpp>
#include <dm.hpp>

using namespace deepmon;

namespace deepmon {
    DM_Blob::DM_Blob(std::vector<uint32_t> shapes, ENVIRONMENT_TYPE evn,
                     PRESICION_TYPE precision_type, float *initialized_data) {
        this->cpu_data = NULL;
        this->gpu_data = NULL;
        this->size = 1;
        for(std::vector<uint32_t >::iterator it = shapes.begin() ; it != shapes.end() ; it++) {
            this->shapes.push_back(*it);
            this->size *= *it;
        }
        this->environment = evn;
        this->precision = precision_type;

        DeepMon::Get().AllocateMemory(this->environment, this, initialized_data);
    }

    DM_Blob::~DM_Blob() {
        if(environment == ENVIRONMENT_CPU) {
            if(this->cpu_data != NULL)
                delete this->cpu_data;
            this->cpu_data = NULL;
        } else {
            if(this->gpu_data != NULL)
                clReleaseMemObject(this->gpu_data);
            this->gpu_data = NULL;
        }
    }

    DM_Blob* DM_Blob::ConvertToCpuBlob() {
        if(this->is_corrupted())
            return NULL;

        DM_Blob *result = DeepMon::Get().ConvertBlob(this, ENVIRONMENT_CPU, PRECISION_32);

        return result;
    }

    DM_Blob *DM_Blob::CovnertToGpuBlob(PRESICION_TYPE precision) {
        if(this->is_corrupted())
            return NULL;

        DM_Blob *result = DeepMon::Get().ConvertBlob(this, ENVIRONMENT_GPU, precision);

        return result;
    }
}

