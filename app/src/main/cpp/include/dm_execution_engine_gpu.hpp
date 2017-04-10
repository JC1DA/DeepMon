/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   dm_gpu.hpp
 * Author: JC1DA
 *
 * Created on March 1, 2017, 4:59 PM
 */

#ifndef DM_GPU_HPP
#define DM_GPU_HPP

#include "dm_common.hpp"
#include "dm_blob.hpp"
#include "dm_execution_engine.hpp"
#include "dm_kernel_defs.hpp"
#include "dm_kernel_object.hpp"
#include <map>
#include <string>

using namespace std;

namespace deepmon {
    class DM_Execution_Engine_GPU : public DM_Execution_Engine {
    private:
        std::vector<std::string> kernel_files {
                std::string("common.cl"),
                std::string("im2col.cl"),
                std::string("conv.cl"),
                std::string("pooling.cl")
        };
        bool has_working_gpu = false;
        bool support_fp16 = false;
        //OpenCL objects
        std::string platform_name;
        uint num_compute_units = 0;
        uint num_queues = 0;
        cl_platform_id platform_id;
        cl_context context;
        cl_device_id device;
        cl_command_queue *queues = NULL;
        cl_program program_32;
        cl_program program_16;

        std::string read_file(std::string path);
        bool scan_for_gpus();
        bool compile_kernels();
        bool compile_kernels(std::string package_path);
        bool read_kernels();
        cl_program build_program(std::string source, std::string build_args);
        std::string get_program_build_log(cl_program program);

        bool read_data_from_host_fp32(cl_mem cl_data, float *data, int size_in_bytes);
        bool read_data_from_host_fp16(cl_mem cl_data, float *data, int size_in_bytes);
        bool read_data_from_host(cl_mem cl_data, float *data, int size_in_bytes, PRESICION_TYPE type);
        DM_Blob *convert_to_gpu_fp32_blob(DM_Blob *blob);
        DM_Blob *convert_to_gpu_fp16_blob(DM_Blob *blob);
        DM_Blob *convert_to_cpu_blob(DM_Blob *blob);

        //kernel activation
        bool execute_memcpy(PRESICION_TYPE precision, cl_mem cl_output, cl_mem cl_input, int num_items);
        bool execute_float_to_half_conversion(cl_mem cl_output, cl_mem cl_input, int num_items);
        bool execute_half_to_float_conversion(cl_mem cl_output, cl_mem cl_input, int num_items);

        //kernels list
        std::vector<std::string> kernel_names {
                std::string(KERNEL_CONVERT_FLOAT_TO_HALF),
                std::string(KERNEL_CONVERT_HALF_TO_FLOAT),
                std::string(KERNEL_MEMCPY),
                std::string(KERNEL_CAFFE_IM2COL),
                std::string(KERNEL_CAFFE_COL2IM),
                std::string(KERNEL_DM_CONV_BASE),
                std::string(KERNEL_CAFFE_MAXPOOL),
                std::string(KERNEL_CAFFE_AVEPOOL)
        };
        std::map<std::string, DM_Kernel_Object *> kernels_map_fp32;
        std::map<std::string, DM_Kernel_Object *> kernels_map_fp16;
    public:
        DM_Execution_Engine_GPU();
        DM_Execution_Engine_GPU(std::string package_path);

        void ExecuteIm2Col(MEMORY_LAYOUT mem_layout, PRESICION_TYPE precision,
                           DM_Blob *input, uint32_t input_offset,
                           uint32_t filter_h, uint32_t filter_w,
                           uint32_t stride_h, uint32_t stride_w,
                           uint32_t pad_left, uint32_t pad_top, uint32_t pad_right, uint32_t pad_bottom,
                           uint32_t dilation_h, uint32_t dilation_w,
                           uint32_t output_h, uint32_t output_w,
                           DM_Blob *im2col_output, uint32_t im2col_offset);


        void FinalizeAllTasks();
        void AllocateMemory(DM_Blob *blob, float *initialized_data);
        DM_Blob *blob_convert_to_cpu_blob(DM_Blob *blob);
        DM_Blob *blob_convert_to_gpu_blob(DM_Blob *blob, PRESICION_TYPE precision);
        cl_command_queue GetCurrentQueue() {
            return this->queues[0];
        }
        cl_context  GetContext() {
            return this->context;
        }

        /*
         * Fixme: this should not be public function
         */
        cl_kernel GetKernel(PRESICION_TYPE precision, string kernel_name) {
            cl_kernel kernel = NULL;
            if(precision == PRECISION_32) {
                kernel = kernels_map_fp32.find(kernel_name)->second->get_kernel();
            } else if(precision == PRECISION_16) {
                kernel = kernels_map_fp16.find(kernel_name)->second->get_kernel();
            }

            return kernel;
        }
    };
}


#endif /* DM_GPU_HPP */

