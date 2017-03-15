#ifndef DM_NET_PARAMETER_HPP
#define DM_NET_PARAMETER_HPP

#include <map>
#include "dm_common.hpp"
#include "dm_layer_param.hpp"
#include <json/json.h>

#include <fstream>

using namespace std;
namespace deepmon {
    class DM_Net_Parameter {
    private:
        bool use_dm_layout = false;
        uint32_t num_layers = -1;
        vector<string> layer_names;
        map<string, DM_Layer_Param *> layer_names_to_layer_params;
        bool failed_to_read = false;
    public:
        DM_Net_Parameter(string net_dir_path) {
            string main_file_path = net_dir_path + "/main.dm";

            ifstream in(main_file_path.c_str());
            Json::Value net;
            in >> net;

            this->use_dm_layout = net["USE_DM_LAYOUT"].asBool();

            for (Json::Value::iterator it = net["LAYERS"].begin(); it != net["LAYERS"].end(); ++it) {
                string name((*it)["name"].asString());
                string type((*it)["type"].asString());
                string conf_path((*it)["conf_file"].asString());
                string w_path((*it)["weights_file"].asString());

                vector<string> inputs;
                for (Json::Value::iterator input_it = (*it)["inputs"].begin(); input_it != (*it)["inputs"].end(); ++input_it) {
                    inputs.push_back((*input_it).asString());
                }

                /*
                 * FixMe: Please check correct type
                 */

                layer_names.push_back(name);

                DM_Layer_Param * layer_param = new DM_Layer_Param(name, type, net_dir_path, conf_path, w_path, inputs, use_dm_layout);

                pair<string, DM_Layer_Param*> pair(name, layer_param);
                layer_names_to_layer_params.insert(pair);
            }

            this->num_layers = this->layer_names.size();

            //the first layer has to be data layer
            if(layer_names_to_layer_params.find(this->layer_names.at(0))->second->GetType().compare(INPUT_NAME)) {
                LOGD("First Layer has to be Data layer");
                failed_to_read = true;
            }
        }
        uint32_t GetNumLayers() {
            return this->num_layers;
        }
        vector<string> GetLayerNames() {
            return vector<string>(layer_names);
        }
        DM_Layer_Param &GetLayerParam(string layer_name) {
            return *(layer_names_to_layer_params.find(layer_name)->second);
        }
        bool IsCorrupted() {
            if(failed_to_read)
                LOGE("Network is corrupted");
            return failed_to_read;
        }
        void PrintNet() {
            if(!IsCorrupted()) {
                LOGD("Network");
                LOGD("\tNumber of layers: %d", num_layers);
                for (int i = 0; i < num_layers; i++) {
                    layer_names_to_layer_params.find(layer_names.at(i))->second->PrintLayerParam();
                }
            }
        }
    };
}

#endif
