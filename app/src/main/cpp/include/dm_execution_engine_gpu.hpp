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
#include <string>
#include <CL/cl.h>
#include <vector>
#include <map>

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

    //kernels list
    std::vector<std::string> kernel_names {
            std::string("convertFloatToHalf"),
            std::string("convertHalfToFloat"),
            std::string("caffe_im2col"),
            std::string("caffe_col2im"),
            std::string("dm_conv_base")
    };
    std::map<std::string, DM_Kernel_Object *> kernels_map_fp32;
    std::map<std::string, DM_Kernel_Object *> kernels_map_fp16;
public:
    DM_Execution_Engine_GPU();
    DM_Execution_Engine_GPU(std::string package_path);
};


#endif /* DM_GPU_HPP */

