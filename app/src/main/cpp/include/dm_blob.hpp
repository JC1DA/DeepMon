#ifndef DM_BLOB_HPP
#define DM_BLOB_HPP

#include <vector>
#include <CL/cl.h>
#include "dm_common.hpp"
#include "dm_log.hpp"

#define ARM_COMPUTE_CL /* So that OpenCL exceptions get caught too */

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
//#include "arm_compute/tests/Utils.h"

#include <arm_compute/core/Helpers.h>
#include <arm_compute/core/ITensor.h>
//#include <arm_compute/core/Validate.h>
#include <arm_compute/runtime/Tensor.h>

using namespace arm_compute;

namespace deepmon {
    class DM_Blob {
    private:
        std::vector<uint32_t> shapes;
        ENVIRONMENT_TYPE environment;
        PRESICION_TYPE precision;

        uint32_t size; //number of items
        uint32_t mem_size; //number of bytes
        bool corrupted = false;

        float *cpu_data;
        cl_mem gpu_data;

        //add ACL Tensor
        CLTensor tensor;

        bool is_persitent = false; //cannot be deleted during forward executions

    public:
        DM_Blob(std::vector<uint32_t> shapes, ENVIRONMENT_TYPE evn, PRESICION_TYPE precision_type, float * initialized_data);
        ~DM_Blob();
        ENVIRONMENT_TYPE get_env() {
            return this->environment;
        }
        PRESICION_TYPE get_precision() {
            return this->precision;
        }
        std::vector<uint32_t> get_shapes() {
            return std::vector<uint32_t>(this->shapes);
        }
        std::vector<uint32_t> get_reverse_shapes() {
            std::vector<uint32_t> v;
            std::vector<uint32_t>::iterator it;
            for(unsigned int i = shapes.size() - 1 ; i >= 0 ; i--) {
                it = v.insert ( it , shapes[i] );
            }
            return v;
        }
        uint32_t get_size() {
            return this->size;
        }
        uint32_t get_mem_size() {
            return this->mem_size;
        }
        void set_size(int size) {
            this->size = size;
        }
        void set_mem_size(int size) {
            this->mem_size = size;
        }
        bool is_corrupted() {
            return this->corrupted;
        }
        void set_corrupted(bool is_corrupted) {
            this->corrupted = is_corrupted;
        }
        float* get_cpu_data() {
            return this->cpu_data;
        }
        void set_cpu_data(float *data) {
            this->cpu_data = data;
        }
        cl_mem get_gpu_data() {
            return this->gpu_data;
        }
        void set_gpu_data(cl_mem data) {
            this->gpu_data = data;
        }
        CLTensor * get_CLTensor() {
            return &tensor;
        }
        void set_persistent(bool is_persistent) {
            this->is_persitent = is_persistent;
        }
        bool is_persistent_blob() {
            return this->is_persitent;
        }
        uint32_t get_shape_at(int idx) {
            if(idx < shapes.size())
                return shapes.at(idx);
            else
                return 1;
        }
        uint32_t get_total_size() {
            uint32_t size = 0;

            uint32_t tmp = 1;
            for(int i = 0 ; i < shapes.size() ; i++) {
                tmp *= shapes.at(i);
            }

            if(shapes.size() != 0)
                size = tmp;

            return size;
        }
        void print_blob() {
            if(this->environment != ENVIRONMENT_CPU) {
                LOGE("Cannot print Non-CPU blob");
                return;
            }

            if(this->shapes.size() == 4) {
                for(int i = 0 ; i < shapes.at(0) ; i++) {
                    for(int j = 0 ; j < shapes.at(1) ; j++) {
                        for(int k = 0 ; k < shapes.at(2) ; k++) {
                            for(int z = 0 ; z < shapes.at(3) ; z++) {
                                int idx = ((i * shapes.at(1) + j) * shapes.at(2) + k) * shapes.at(3) + z;
                                LOGD("[%d,%d,%d,%d]: %f", i, j, k, z, cpu_data[idx]);
                            }
                        }
                    }
                }
            } else if(this->shapes.size() == 3) {
                for(int i = 0 ; i < shapes.at(0) ; i++) {
                    for(int j = 0 ; j < shapes.at(1) ; j++) {
                        for(int k = 0 ; k < shapes.at(2) ; k++) {
                            int idx = ((i * shapes.at(1) + j) * shapes.at(2) + k);
                            LOGD("[%d,%d,%d]: %f", i, j, k, cpu_data[idx]);
                        }
                    }
                }
            } else if(this->shapes.size() == 2) {
                for(int i = 0 ; i < shapes.at(0) ; i++) {
                    for(int j = 0 ; j < shapes.at(1) ; j++) {
                        int idx = (i * shapes.at(1) + j);
                        LOGD("[%d,%d]: %f", i, j, cpu_data[idx]);
                    }
                }
            } else if(this->shapes.size() == 1) {
                for(int i = 0 ; i < shapes.at(0) ; i++) {
                    LOGD("[%d]: %f", i, cpu_data[i]);
                }
            }
        }
        DM_Blob *ConvertToCpuBlob();
        DM_Blob *CovnertToGpuBlob(PRESICION_TYPE precision);
    };

}

#endif
