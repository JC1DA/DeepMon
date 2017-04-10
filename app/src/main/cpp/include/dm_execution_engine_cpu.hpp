#ifndef DM_EXECUTION_ENGINE_CPU_HPP
#define DM_EXECUTION_ENGINE_CPU_HPP


#include <cstdint>
#include <vector>
#include "dm_common.hpp"
#include "dm_execution_engine.hpp"
#include "dm_blob.hpp"

namespace deepmon {

    class DM_Execution_Engine_CPU : public DM_Execution_Engine {
    public:
        DM_Execution_Engine_CPU();
        void AllocateMemory(DM_Blob *blob, float *initialized_data);
    };
}

#endif
