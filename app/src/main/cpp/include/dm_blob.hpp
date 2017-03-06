#ifndef DM_BLOB_HPP
#define DM_BLOB_HPP

#include <vector>
#include <CL/cl.h>
#include "dm_common.hpp"

class DM_Blob {
private:
    std::vector<int> shapes;
    ENVIRONMENT_TYPE environment;
    PRESICION_TYPE precision;

    int size; //number of items
    int mem_size; //number of bytes
    bool is_corrupted = false;

    char *cpu_data;
    cl_mem gpu_data;

    void create_Cpu_mem(char * initialized_data);
    void create_Gpu_mem(PRESICION_TYPE type, char * initialized_data);
public:
    DM_Blob(std::vector<int> shapes, ENVIRONMENT_TYPE evn, PRESICION_TYPE precision_type, char * initialized_data);
};

#endif
