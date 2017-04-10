#ifndef DM_KERNEL_OBJECT
#define DM_KERNEL_OBJECT

#include "CL/cl.h"

namespace deepmon {
    class DM_Kernel_Object {
    private:
        cl_kernel kernel;

    public:
        DM_Kernel_Object(cl_kernel kernel) {
            this->kernel = kernel;
        }
        cl_kernel get_kernel() {
            return this->kernel;
        }
    };
}

#endif
