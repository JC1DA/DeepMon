#ifndef DM_NET_HPP
#define DM_NET_HPP

#include "dm_net_parameter.hpp"
#include "dm_layer.hpp"

using namespace std;
namespace deepmon {
    class DM_Net {
    private:
        vector<DM_Layer *> layers;
        map<string, DM_Layer *> name_to_layer_map;
        vector<DM_Layer *> pipeline;
        bool is_working = true;
    protected:
    public:
        DM_Net(string model_dir_path);

        DM_Blob *Forward(float *data);

        bool IsWorking() {
            if(!is_working)
                LOGE("Network is corrupted");
            return is_working;
        }

        void PrintNet() {
            if(IsWorking()) {
                LOGD("Network");
                LOGD("\tNumber of layers: %d", layers.size());
                for (int i = 0; i < layers.size(); i++) {
                    name_to_layer_map.find(layers.at(i)->GetName())->second->PrintInfo();
                }
                for(int i = 0 ; i < this->layers.size() ; i++)
                    this->layers.at(i)->PrintInfo();
            }
        }

        void PrintProcessingPileline() {
            if(IsWorking()) {
                LOGD("Processing pipeline:");
                for(int i = 0 ; i < pipeline.size() ; i++) {
                    LOGD("\t%s", pipeline.at(i)->GetName().c_str());
                }
            }
        }
    };
}

#endif
