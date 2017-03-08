/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "dm_configs.hpp"
#include "dm_log.hpp"
#include "dm_err.hpp"
#include "dm_execution_engine.hpp"
#include "dm_execution_engine_gpu.hpp"
#include "dm_kernels.hpp"
#include "dm_kernel_object.hpp"
#include "dm_kernel_defs.hpp"

#include <sys/stat.h>
#include <string>

namespace deepmon {
    DM_Execution_Engine_GPU::DM_Execution_Engine_GPU() : DM_Execution_Engine(ENVIRONMENT_GPU) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif

        //initialize GPU
        if(!this->scan_for_gpus()) {
            LOGE("Failed to scan for gpus");
            return;
        }

        //compile kernels
        if(!compile_kernels()) {
            LOGE("Failed to compile kernel code");
            return;
        }

        //extract kernels
        if(!read_kernels()) {
            LOGE("Failed to extract kernels from program code");
            return;
        }

        this->has_working_gpu = true;
        this->initialized = true;
    }

    DM_Execution_Engine_GPU::DM_Execution_Engine_GPU(std::string package_path) : DM_Execution_Engine(ENVIRONMENT_GPU) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif

        //initialize GPU
        if(!this->scan_for_gpus()) {
            LOGE("Failed to scan for gpus");
            return;
        }

        //compile kernels
        if(!compile_kernels(package_path)) {
            LOGE("Failed to compile kernel code");
            return;
        }

        //extract kernels
        if(!read_kernels()) {
            LOGE("Failed to extract kernels from program code");
            return;
        }

        this->has_working_gpu = true;
        this->initialized = true;
    }

    bool DM_Execution_Engine_GPU::scan_for_gpus() {
        bool initialized = false;

#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif

#ifdef PRINT_STEPS
        LOGD("Scanning for GPUs");
#endif

        cl_int err = CL_SUCCESS;

        cl_uint num_of_platforms = 0;
        err = clGetPlatformIDs(0, 0, &num_of_platforms);
        SAMPLE_CHECK_ERRORS_WITH_FALSE_RETURN(err);

        if(num_of_platforms < 1) {
            return initialized;
        }

#ifdef PRINT_VARS
        LOGD("Found %d platforms", num_of_platforms);
#endif

        cl_platform_id *platforms = new cl_platform_id[num_of_platforms];
        err = clGetPlatformIDs(num_of_platforms, platforms, 0);
        SAMPLE_CHECK_ERRORS_WITH_FALSE_RETURN(err);

#ifdef PRINT_VARS
        for(int idx = 0 ; idx < num_of_platforms ; idx++)
            LOGD("Found platform_id = %x", platforms[idx]);
#endif

        for (int idx = 0; idx < num_of_platforms; idx++) {
            this->platform_id = platforms[idx];

            size_t platform_name_length = 0;
            err = clGetPlatformInfo(
                    platforms[idx],
                    CL_PLATFORM_NAME,
                    0,
                    0,
                    &platform_name_length
            );
            SAMPLE_CHECK_ERRORS(err);
            if (err != CL_SUCCESS) continue;

            char *platform_name = new char[platform_name_length + 1];
            platform_name[platform_name_length] = '\0';

            err = clGetPlatformInfo(
                    platforms[idx],
                    CL_PLATFORM_NAME,
                    platform_name_length,
                    platform_name,
                    0
            );
            SAMPLE_CHECK_ERRORS(err);
            if (err != CL_SUCCESS) {
                delete platform_name;
                continue;
            }

            this->platform_name.assign(platform_name);
            delete platform_name;

#ifdef PRINT_VARS
            LOGD("Found platform with name: %s", this->platform_name.c_str());
#endif

            //query for context
            cl_context_properties context_props[] = {
                    CL_CONTEXT_PLATFORM,
                    cl_context_properties(platforms[idx]),
                    0
            };

            this->context = clCreateContextFromType(
                    context_props,
                    CL_DEVICE_TYPE_GPU,
                    0,
                    0,
                    &err
            );
            SAMPLE_CHECK_ERRORS(err);
            if (err != CL_SUCCESS) continue;

            err = clGetContextInfo(
                    this->context,
                    CL_CONTEXT_DEVICES,
                    sizeof (this->device),
                    &this->device,
                    0);
            SAMPLE_CHECK_ERRORS(err);
            if (err != CL_SUCCESS) continue;

            //get number of compute units
            err = clGetDeviceInfo(
                    this->device,
                    CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof (this->num_compute_units),
                    &this->num_compute_units,
                    0);
            SAMPLE_CHECK_ERRORS(err);
            if (err != CL_SUCCESS) continue;

#ifdef PRINT_VARS
            LOGD("Found %d compute units on %s", this->num_compute_units, this->platform_name.c_str());
#endif

            char version[32]; version[31] = '\0';
            size_t version_size = 0;
            clGetDeviceInfo(this->device,
                            CL_DEVICE_VERSION,
                            sizeof (version),
                            version,
                            &version_size);

#ifdef PRINT_VARS
            LOGD("Device has version %s", version);
#endif

            //create one queue for each compute units just in case
            this->queues = new cl_command_queue[this->num_compute_units];
            for(uint qid = 0 ; qid < this->num_compute_units ; qid++) {
                this->queues[qid] = clCreateCommandQueue(
                        this->context,
                        this->device,
                        0, // Creating queue properties, refer to the OpenCL specification for details.
                        &err);
                SAMPLE_CHECK_ERRORS(err);
                if (err != CL_SUCCESS) {
                    break;
                } else
                    this->num_queues++;
            }

            if(this->num_queues == 0) {
                this->num_compute_units = 0;
                delete this->queues;
                continue;
            }

#ifdef PRINT_VARS
            LOGD("Created %d queues for executions", this->num_queues);
#endif
            initialized = true;
            break;
        }

        delete platforms;
        return initialized;
    }

    cl_program DM_Execution_Engine_GPU::build_program(std::string source, std::string build_args) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif

        cl_int err = CL_SUCCESS;
        const char *kernelSource = source.c_str();
        cl_program program = clCreateProgramWithSource(
                this->context,
                1,
                &kernelSource,
                0,
                &err
        );
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        err = clBuildProgram(program, 0, 0, build_args.c_str(), 0, 0);
        SAMPLE_CHECK_ERRORS(err);
        if(err == CL_BUILD_PROGRAM_FAILURE) {
            std::string build_log = get_program_build_log(program);
            LOGD("BUILDING FAILED: %s", build_log.c_str());
            return NULL;
        }

        return program;
    }

    std::string DM_Execution_Engine_GPU::get_program_build_log(cl_program program) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif

        cl_int err = CL_SUCCESS;
        size_t log_length = 0;
        std::string log("");

        err = clGetProgramBuildInfo(
                program,
                this->device,
                CL_PROGRAM_BUILD_LOG,
                0,
                0,
                &log_length);
        SAMPLE_CHECK_ERRORS(err);
        if(err != CL_SUCCESS) return log;

        char* log_buf = new char[log_length + 1]; log_buf[log_length] = '\0';
        err = clGetProgramBuildInfo(
                program,
                this->device,
                CL_PROGRAM_BUILD_LOG,
                log_length,
                (void*) log_buf,
                0);
        SAMPLE_CHECK_ERRORS(err);
        if(err != CL_SUCCESS) {
            delete log_buf;
            return log;
        }

        log.append(log_buf);
        delete log_buf;
        return log;
    }

    bool DM_Execution_Engine_GPU::compile_kernels() {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif

        bool is_compiled = true;
        cl_int err = CL_SUCCESS;

        std::string build_args = "";

        //compile for 32-bits floating point
        std::string source_string = "";
        source_string += "#define PRECISION 32\n";
        for(int i = 0 ; i < kernels.size() ; i++) {
            source_string += kernels.at(i);
        }

        //LOGD("%s",source_string.c_str());

        cl_program program_32 = build_program(source_string, build_args);
        if(program_32 == NULL)
            return false;
        this->program_32 = program_32;

        //compile for 16-bits floating point
        char extensions[1024];
        size_t extension_length = 0;
        clGetDeviceInfo(this->device, CL_DEVICE_EXTENSIONS,
                        sizeof (extensions),
                        extensions,
                        &extension_length);

        //search support for fp16 in extensions
        if(std::string(extensions).find(std::string("cl_khr_fp16")) != std::string::npos) {
            source_string = "";
            source_string += "#define PRECISION 16\n";
            for(int i = 0 ; i < kernels.size() ; i++) {
                source_string += kernels.at(i);
            }
            cl_program program_16 = build_program(source_string, build_args);
            if(program_16 != NULL) {
                this->support_fp16 = true;
                this->program_16 = program_16;
            }
        } else {
            LOGD("No support for cl_khr_fp16");
        }

        return is_compiled;
    }

    extern "C"
    std::string DM_Execution_Engine_GPU::read_file(std::string path) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif

        FILE *fp = fopen(path.c_str(),"r");
        int fd = fileno(fp);
        struct stat buf;
        fstat(fd, &buf);
        int size = buf.st_size;

        char *buffer = new char[size + 1];
        buffer[size] = '\0';
        fread(buffer, size, 1, fp);
        fclose(fp);

        std::string data(buffer);
        delete buffer;

        return data;
    }

    bool DM_Execution_Engine_GPU::compile_kernels(std::string package_path) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif

        bool is_compiled = true;
        cl_int err = CL_SUCCESS;

        std::string build_args = "";

        //compile for 32-bits floating point
        std::string source_string = "";
        source_string += "#define PRECISION 32\n";
        for(int i = 0 ; i < this->kernel_files.size() ; i++) {
            //read file
            std::string data = read_file(package_path + "/" + this->kernel_files.at(i));
            source_string += data + "\n";
        }

        cl_program program_32 = build_program(source_string, build_args);
        if(program_32 == NULL)
            return false;
        this->program_32 = program_32;

        //compile for 16-bits floating point
        char extensions[1024];
        size_t extension_length = 0;
        clGetDeviceInfo(this->device, CL_DEVICE_EXTENSIONS,
                        sizeof (extensions),
                        extensions,
                        &extension_length);

        //search support for fp16 in extensions
        if(std::string(extensions).find(std::string("cl_khr_fp16")) != std::string::npos) {
            source_string = "";
            source_string += "#define PRECISION 16\n";
            for(int i = 0 ; i < this->kernel_files.size() ; i++) {
                //read file
                std::string data = read_file(package_path + "/" + this->kernel_files.at(i));
                source_string += data + "\n";
            }
            cl_program program_16 = build_program(source_string, build_args);
            if(program_16 != NULL) {
                this->support_fp16 = true;
                this->program_16 = program_16;
            }
        } else {
            LOGD("No support for cl_khr_fp16");
        }

        return is_compiled;
    }

    bool DM_Execution_Engine_GPU::read_kernels() {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
        bool is_successful = false;

        if(this->program_32 != NULL) {
            cl_int err = CL_SUCCESS;
            for(int i = 0 ; i < this->kernel_names.size() ; i++) {
                std::string kernel_name = this->kernel_names.at(i);
#ifdef PRINT_VARS
                LOGD("FP32 Program: Extracting %s kernel", kernel_name.c_str());
#endif
                cl_kernel kernel = clCreateKernel(this->program_32, kernel_name.c_str(), &err);
                SAMPLE_CHECK_ERRORS(err);
                if(err == CL_SUCCESS) {
                    //extract more information
                    DM_Kernel_Object *kobj = new DM_Kernel_Object(kernel);
                    std::pair<std::string, DM_Kernel_Object *> pair(kernel_name, kobj);
                    this->kernels_map_fp32.insert(pair);
                }
            }
            is_successful = true;
        }

        if(this->program_16 != NULL) {
            cl_int err = CL_SUCCESS;
            for(int i = 0 ; i < this->kernel_names.size() ; i++) {
                std::string kernel_name = this->kernel_names.at(i);
#ifdef PRINT_VARS
                LOGD("FP16 Program: Extracting %s kernel", kernel_name.c_str());
#endif
                cl_kernel kernel = clCreateKernel(this->program_16, kernel_name.c_str(), &err);
                SAMPLE_CHECK_ERRORS(err);
                if(err == CL_SUCCESS) {
                    //extract more information
                    DM_Kernel_Object *kobj = new DM_Kernel_Object(kernel);
                    std::pair<std::string, DM_Kernel_Object *> pair(kernel_name, kobj);
                    this->kernels_map_fp16.insert(pair);
                }
            }
        }

        return is_successful;
    }

    cl_command_queue DM_Execution_Engine_GPU::get_current_queue() {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
        /*static int current_idx = 0;
        cl_command_queue queue = this->queues[current_idx];
        current_idx = (current_idx + 1) % this->num_queues;
        return queue;*/
        return this->queues[0];
    }

    void DM_Execution_Engine_GPU::finalize_all_tasks() {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
        for(int i = 0 ; i < this->num_queues ; i++) {
            cl_int err = clFinish(this->queues[i]);
            SAMPLE_CHECK_ERRORS(err);
        }
    }

    bool DM_Execution_Engine_GPU::read_data_from_host_fp32(cl_mem cl_data, float *data, int size_in_bytes) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
        cl_int err = CL_SUCCESS;

        cl_command_queue current_queue = get_current_queue();

        float *buf_dst = (float *)clEnqueueMapBuffer(current_queue, \
					cl_data, \
					CL_TRUE, CL_MAP_WRITE, \
					0, \
					size_in_bytes, \
					0, NULL, NULL, &err);
        SAMPLE_CHECK_ERRORS(err);
        if(err != CL_SUCCESS) {
            return false;
        }

        memcpy((void*)buf_dst, (void*)data, size_in_bytes);

        clEnqueueUnmapMemObject(current_queue, \
					cl_data, \
					buf_dst, \
					0, NULL, NULL);

        return true;
    }

    bool DM_Execution_Engine_GPU::read_data_from_host_fp16(cl_mem cl_data, float *data,
                                                           int size_in_bytes) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
        cl_int err = CL_SUCCESS;

        if(!this->support_fp16)
            return false;

        cl_mem cl_tmp = clCreateBuffer(
                this->context,
                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                size_in_bytes * 2, //size in bytes * 2 because of float
                data,//buffer of data
                &err);
        SAMPLE_CHECK_ERRORS(err);
        if(err != CL_SUCCESS)
            return false;

        cl_kernel kernel = (this->kernels_map_fp16.find(std::string(KERNEL_CONVERT_FLOAT_TO_HALF))->second)->get_kernel();
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_tmp);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_data);
        SAMPLE_CHECK_ERRORS(err);
        if(err != CL_SUCCESS)
            return false;

        size_t wgs[1] = {(size_t) size_in_bytes / sizeof(cl_half)};

        cl_command_queue queue = get_current_queue();
        err = clEnqueueNDRangeKernel(
                queue,
                kernel,
                1,
                0,
                wgs,
                0,
                0, 0, 0
        );
        SAMPLE_CHECK_ERRORS(err);
        if(err != CL_SUCCESS)
            return false;

        err = clFinish(queue);
        SAMPLE_CHECK_ERRORS(err);

        clReleaseMemObject(cl_tmp);

        return true;
    }

    bool DM_Execution_Engine_GPU::read_data_from_host(cl_mem cl_data, float *data,
                                                      int size_in_bytes, PRESICION_TYPE type) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
        if(type == PRECISION_32)
            return read_data_from_host_fp32(cl_data, data, size_in_bytes);
        if(type == PRECISION_16 && this->support_fp16)
            return read_data_from_host_fp16(cl_data, data, size_in_bytes);

        return false;
    }

    DM_Blob * DM_Execution_Engine_GPU::convert_to_cpu_blob(DM_Blob *blob) {
        DM_Blob *result = NULL;

        if(blob->get_env() == ENVIRONMENT_CPU) {
            result = new DM_Blob(blob->get_shapes(), blob->get_env(), blob->get_precision(), blob->get_cpu_data());
        } else if(blob->get_env() == ENVIRONMENT_GPU) {
            if(!this->has_working_gpu)
                return NULL;
            if(blob->get_precision() == PRECISION_16 && !this->support_fp16)
                return NULL;

            cl_int err = CL_SUCCESS;

            cl_mem cl_data = NULL;
            int cl_data_size = blob->get_size() * sizeof(cl_float);
            if(blob->get_precision() == PRECISION_16) {
                cl_data = clCreateBuffer(
                        this->context,
                        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                        cl_data_size, //size in bytes * 2 because of float
                        NULL,//buffer of data
                        &err);
                SAMPLE_CHECK_ERRORS(err);
                if(err != CL_SUCCESS)
                    return NULL;

                if(!execute_half_to_float_conversion(cl_data, blob->get_gpu_data(), blob->get_size())) {
                    clReleaseMemObject(cl_data);
                    return NULL;
                }
            } else if(blob->get_precision() == PRECISION_32) {
                cl_data = blob->get_gpu_data();
            }

            cl_command_queue current_queue = get_current_queue();

            float *data = (float *)clEnqueueMapBuffer(current_queue, \
					cl_data, \
					CL_TRUE, CL_MAP_READ, \
					0, \
					cl_data_size, \
					0, NULL, NULL, &err);
            SAMPLE_CHECK_ERRORS(err);
            if(err != CL_SUCCESS) {
                if(blob->get_precision() == PRECISION_16)
                    clReleaseMemObject(cl_data);
                return NULL;
            }

            result = new DM_Blob(blob->get_shapes(), ENVIRONMENT_CPU, PRECISION_32, data);

            clEnqueueUnmapMemObject(current_queue, \
					cl_data, \
					data, \
					0, NULL, NULL);

            if(blob->get_precision() == PRECISION_16)
                clReleaseMemObject(cl_data);
        }

        return result;
    }

    DM_Blob * DM_Execution_Engine_GPU::convert_to_gpu_fp32_blob(DM_Blob *blob) {
        DM_Blob *result = NULL;

        if(!this->has_working_gpu)
            return NULL;

        if(blob->get_env() == ENVIRONMENT_GPU) {
            if(blob->get_precision() == PRECISION_16 && !this->support_fp16)
                return NULL;

            result = new DM_Blob(blob->get_shapes(), ENVIRONMENT_GPU, PRECISION_32, NULL);

            bool is_successful;
            if(blob->get_precision() == PRECISION_32)
                is_successful = execute_memcpy(PRECISION_32, result->get_gpu_data(), blob->get_gpu_data(), blob->get_size());
            else if(blob->get_precision() == PRECISION_16)
                is_successful = execute_half_to_float_conversion(result->get_gpu_data(), blob->get_gpu_data(), blob->get_size());

            if(!is_successful) {
                delete result;
                return NULL;
            }
        } else if(blob->get_env() == ENVIRONMENT_CPU) {
            result = new DM_Blob(blob->get_shapes(), ENVIRONMENT_GPU, PRECISION_32, blob->get_cpu_data());
        }

        return result;
    }

    DM_Blob * DM_Execution_Engine_GPU::convert_to_gpu_fp16_blob(DM_Blob *blob) {
        if(!this->has_working_gpu || !this->support_fp16)
            return NULL;

        DM_Blob *result = NULL;

        if(blob->get_env() == ENVIRONMENT_GPU) {
            result = new DM_Blob(blob->get_shapes(), ENVIRONMENT_GPU, PRECISION_16, NULL);

            bool is_successful;
            if(blob->get_precision() == PRECISION_16)
                is_successful = execute_memcpy(PRECISION_16, result->get_gpu_data(), blob->get_gpu_data(), blob->get_size());
            else if(blob->get_precision() == PRECISION_32)
                is_successful = execute_float_to_half_conversion(result->get_gpu_data(), blob->get_gpu_data(), blob->get_size());

            if(!is_successful) {
                delete result;
                return NULL;
            }
        } else if(blob->get_env() == ENVIRONMENT_CPU) {
            result = new DM_Blob(blob->get_shapes(), ENVIRONMENT_GPU, PRECISION_16, blob->get_cpu_data());
        }

        return result;
    }

    bool DM_Execution_Engine_GPU::execute_memcpy(PRESICION_TYPE precision, cl_mem cl_output, cl_mem cl_input,
                                                   int num_items) {
        cl_command_queue current_queue = get_current_queue();
        cl_int err = CL_SUCCESS;

        cl_kernel kernel = NULL;
        if(precision == PRECISION_32)
            kernel = (this->kernels_map_fp32.find(std::string(KERNEL_MEMCPY)))->second->get_kernel();
        else if(precision == PRECISION_16)
            kernel = (this->kernels_map_fp16.find(std::string(KERNEL_MEMCPY)))->second->get_kernel();

        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_input);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_output);
        SAMPLE_CHECK_ERRORS(err);
        if(err != CL_SUCCESS) {
            return false;
        }

        size_t wgs[1] = {(size_t)num_items};

        err = clEnqueueNDRangeKernel(
                current_queue,
                kernel,
                1,
                0,
                wgs,
                0,
                0, 0, 0
        );
        err |= clFinish(current_queue);
        SAMPLE_CHECK_ERRORS(err);

        if(err != CL_SUCCESS) {
            return false;
        }

        return true;
    }

    bool DM_Execution_Engine_GPU::execute_float_to_half_conversion(cl_mem cl_output, cl_mem cl_input,
                                                                     int num_items) {
        cl_command_queue current_queue = get_current_queue();
        cl_int err = CL_SUCCESS;

        cl_kernel kernel = (this->kernels_map_fp16.find(std::string(KERNEL_CONVERT_FLOAT_TO_HALF)))->second->get_kernel();

        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_input);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_output);
        SAMPLE_CHECK_ERRORS(err);
        if(err != CL_SUCCESS) {
            return false;
        }

        size_t wgs[1] = {(size_t)num_items};

        err = clEnqueueNDRangeKernel(
                current_queue,
                kernel,
                1,
                0,
                wgs,
                0,
                0, 0, 0
        );
        err |= clFinish(current_queue);
        SAMPLE_CHECK_ERRORS(err);

        if(err != CL_SUCCESS) {
            return NULL;
        }

        return true;
    }

    bool DM_Execution_Engine_GPU::execute_half_to_float_conversion(cl_mem cl_output, cl_mem cl_input,
                                                                     int num_items) {
        cl_command_queue current_queue = get_current_queue();
        cl_int err = CL_SUCCESS;

        cl_kernel kernel = (this->kernels_map_fp16.find(std::string(KERNEL_CONVERT_HALF_TO_FLOAT)))->second->get_kernel();

        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_input);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_output);
        SAMPLE_CHECK_ERRORS(err);
        if(err != CL_SUCCESS) {
            return false;
        }

        size_t wgs[1] = {(size_t)num_items};

        err = clEnqueueNDRangeKernel(
                current_queue,
                kernel,
                1,
                0,
                wgs,
                0,
                0, 0, 0
        );
        err |= clFinish(current_queue);
        SAMPLE_CHECK_ERRORS(err);

        if(err != CL_SUCCESS) {
            return NULL;
        }

        return true;
    }

    void DM_Execution_Engine_GPU::create_memory(DM_Blob *blob, float *initialized_data) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
        if(!blob->is_corrupted() && blob->get_env() == this->type) {
            int size_in_bytes = 0;
            if(blob->get_precision() == PRECISION_32 && this->has_working_gpu) {
                size_in_bytes = blob->get_size() * sizeof(cl_float);
            } else if(blob->get_precision() == PRECISION_16 && this->support_fp16) {
                size_in_bytes = blob->get_size() * sizeof(cl_half);
            } else {
                blob->set_corrupted(true);
                return;
            }
            blob->set_mem_size(size_in_bytes);

            cl_int err = CL_SUCCESS;
            cl_mem cl_data = clCreateBuffer(
                    this->context,
                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                    size_in_bytes, //size in bytes
                    NULL,//buffer of data
                    &err);
            SAMPLE_CHECK_ERRORS(err);
            if(err != CL_SUCCESS) {
                blob->set_corrupted(true);
                return;
            }

            if(initialized_data != NULL) {
                if(!read_data_from_host(cl_data, initialized_data, size_in_bytes, blob->get_precision())) {
                    blob->set_corrupted(true);
                    return;
                }
            }

            blob->set_gpu_data(cl_data);
        } else
            blob->set_corrupted(true);
    }

    DM_Blob * DM_Execution_Engine_GPU::blob_convert_to_cpu_blob(DM_Blob *blob) {
        return convert_to_cpu_blob(blob);
    }

    DM_Blob * DM_Execution_Engine_GPU::blob_convert_to_gpu_blob(DM_Blob *blob,
                                                                PRESICION_TYPE precision) {
        if(precision == PRECISION_16)
            return convert_to_gpu_fp16_blob(blob);
        else if(precision == PRECISION_32)
            return convert_to_gpu_fp32_blob(blob);
        else
            return NULL;
    }

    void DM_Execution_Engine_GPU::do_conv(DM_Blob *input, DM_Blob *output, DM_Blob *filters,
                                          DM_Blob *biases, std::vector<int> strides,
                                          std::vector<int> pads, std::vector<int> dilations) {

    }
}