#ifndef DM_BLOB_HPP
#define DM_BLOB_HPP

#include <vector>
#include <CL/cl.h>
#include "dm_common.hpp"
#include "dm_log.hpp"

namespace deepmon {
    class DM_Blob {
    private:
        std::vector<uint32_t> shapes;
        ENVIRONMENT_TYPE environment;
        PRESICION_TYPE precision;

        uint32_t refs = 0; //use to track how many layers will use this blobs in forwarding step

        int size; //number of items
        int mem_size; //number of bytes
        bool corrupted = false;

        float *cpu_data;
        cl_mem gpu_data;

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
        int get_shape_at(int idx) {
            if(idx < shapes.size())
                return shapes.at(idx);
            else
                return 1;
        }
        void increase_ref() {
            this->refs++;
        }
        void free() {
            this->refs--;
            if(this->refs < 1) {
                //free this blob
                delete this;
            }
        }
        void print_blob() {
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
    };

}

#endif
