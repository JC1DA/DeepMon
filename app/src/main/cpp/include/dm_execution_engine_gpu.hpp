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

#include "dm_execution_engine.hpp"
#include "dm_kernel_object.hpp"
#include "dm_blob.hpp"
#include <string>
#include <CL/cl.h>
#include <vector>
#include <map>
#include "dm_kernel_defs.hpp"

namespace deepmon {
    class DM_Execution_Engine_GPU : public DM_Execution_Engine {
    private:
        std::vector<std::string> kernel_files {
                std::string("common.cl"),
                std::string("im2col.cl"),
                std::string("conv.cl")
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
        cl_command_queue get_current_queue();

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
                std::string("caffe_im2col"),
                std::string("caffe_col2im"),
                std::string("dm_conv_base")
        };
        std::map<std::string, DM_Kernel_Object *> kernels_map_fp32;
        std::map<std::string, DM_Kernel_Object *> kernels_map_fp16;
    public:
        DM_Execution_Engine_GPU();
        DM_Execution_Engine_GPU(std::string package_path);

        void finalize_all_tasks();
        void create_memory(DM_Blob *blob, float *initialized_data);
        DM_Blob *blob_convert_to_cpu_blob(DM_Blob *blob);
        DM_Blob *blob_convert_to_gpu_blob(DM_Blob *blob, PRESICION_TYPE precision);
        void do_im2col(ENVIRONMENT_TYPE evn_type, MEMORY_LAYOUT mem_layout, DM_Blob *input, DM_Blob *output, \
            std::vector<uint32_t> filters_sizes, std::vector<uint32_t> strides, std::vector<uint32_t> pads, std::vector<uint32_t> dilations);
        void do_conv(MEMORY_LAYOUT mem_layout, DM_Blob *input, DM_Blob *output, \
            DM_Blob *filters, DM_Blob *biases, std::vector<uint32_t> strides, std::vector<uint32_t> pads, std::vector<uint32_t> dilations);
    };
}


#endif /* DM_GPU_HPP */

