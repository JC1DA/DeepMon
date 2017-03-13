//
// Created by JC1DA on 3/13/17.
//

#include <dm.hpp>
#include "dm_net.hpp"

using namespace deepmon;
namespace deepmon {
    DM_Net::DM_Net(string model_dir_path) {
        this->net_param = new DM_Net_Parameter(model_dir_path);
        if(this->net_param->IsCorrupted())
            this->is_working = false;

    }
}