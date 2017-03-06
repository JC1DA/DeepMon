//
// Created by JC1DA on 3/6/17.
//

#include <dm_blob.hpp>
#include <vector>
#include <dm_log.hpp>
#include <string.h>
#include <dm.hpp>
#include "CL/cl.h"
#include <dm_log.hpp>
#include "dm_configs.hpp"

namespace deepmon {
    DM_Blob::DM_Blob(std::vector<int> shapes, ENVIRONMENT_TYPE evn, PRESICION_TYPE precision_type, float * initialized_data) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
        this->size = 1;
        for(std::vector<int>::iterator it = shapes.begin() ; it != shapes.end() ; it++) {
            this->shapes.push_back(*it);
            this->size *= *it;
        }
        this->environment = evn;
        this->precision = precision_type;

        if(this->environment == ENVIRONMENT_CPU) {
            DeepMon::Get().get_excution_engine(true)->create_memory(this, initialized_data);
        } else if(this->environment == ENVIRONMENT_GPU) {
            DeepMon::Get().get_excution_engine(false)->create_memory(this, initialized_data);
        } else {
            LOGE("Unrecognized Environment !!!");
            this->corrupted = true;
        }
    }


}

