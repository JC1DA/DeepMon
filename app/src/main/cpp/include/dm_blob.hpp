#ifndef DM_BLOB_HPP
#define DM_BLOB_HPP

#include <vector>
#include <CL/cl.h>
#include "dm_common.hpp"

namespace deepmon {
    class DM_Blob {
    private:
        std::vector<int> shapes;
        ENVIRONMENT_TYPE environment;
        PRESICION_TYPE precision;

        int size; //number of items
        int mem_size; //number of bytes
        bool corrupted = false;

        float *cpu_data;
        cl_mem gpu_data;

    public:
        DM_Blob(std::vector<int> shapes, ENVIRONMENT_TYPE evn, PRESICION_TYPE precision_type, float * initialized_data);
        ENVIRONMENT_TYPE get_env() {
            return this->environment;
        }
        PRESICION_TYPE get_precision() {
            return this->precision;
        }
        std::vector<int> get_shapes() {
            return std::vector<int>(this->shapes);
        }
        int get_size() {
            return this->size;
        }
        int get_mem_size() {
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
    };

}

#endif
