/*The MIT License (MIT)
 *
 *Copyright (c) 2013 Thomas Park
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *       of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *       to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *       copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *       The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *THE SOFTWARE.
 */

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

