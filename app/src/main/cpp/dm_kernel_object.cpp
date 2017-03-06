//
// Created by JC1DA on 3/6/17.
//

#include <CL/cl.h>
#include "dm_kernel_object.hpp"

namespace deepmon {
    DM_Kernel_Object::DM_Kernel_Object(cl_kernel kernel) {
        this->kernel = kernel;
    }

    cl_kernel DM_Kernel_Object::get_kernel() {
        return this->kernel;
    }
}