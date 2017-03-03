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

#include <sys/stat.h>

DM_Execution_Engine_GPU::DM_Execution_Engine_GPU() : DM_Execution_Engine(ENVIRONMENT_GPU) {
#ifdef PRINT_FUNCTION_NAME
    LOGD("--%s--", __PRETTY_FUNCTION__);
#endif

    //initialize GPU
    if(!this->scan_for_gpus())
        return;
    
    //compile kernels
    if(compile_kernels())
        return;
    
    //extract kernels
    
    
    this->initialized = true;
}

DM_Execution_Engine_GPU::DM_Execution_Engine_GPU(std::string package_path) : DM_Execution_Engine(ENVIRONMENT_GPU) {
#ifdef PRINT_FUNCTION_NAME
    LOGD("--%s--", __PRETTY_FUNCTION__);
#endif

    //initialize GPU
    if(!this->scan_for_gpus())
        return;

    //compile kernels
    if(compile_kernels(package_path))
        return;
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