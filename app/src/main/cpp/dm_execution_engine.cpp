/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "dm_execution_engine.hpp"
#include <dm_log.hpp>
#include "dm_configs.hpp"

namespace deepmon {
    DM_Execution_Engine::DM_Execution_Engine(ENVIRONMENT_TYPE type) {
#ifdef PRINT_FUNCTION_NAME
        LOGD("--%s--", __PRETTY_FUNCTION__);
#endif
        this->type = type;
    }
}