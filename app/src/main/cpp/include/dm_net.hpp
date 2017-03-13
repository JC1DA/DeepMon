#ifndef DM_NET_HPP
#define DM_NET_HPP

#include "dm_net_parameter.hpp"

using namespace std;
namespace deepmon {
    class DM_Net {
    private:
        DM_Net_Parameter *net_param;
        bool is_working = true;
    protected:
    public:
        DM_Net(string model_dir_path);
        bool IsWorking() {
            if(!is_working)
                LOGE("Network is corrupted");
            return is_working;
        }

        void PrintNet() {
            if(IsWorking())
                net_param->PrintNet();
        }
    };
}

#endif
