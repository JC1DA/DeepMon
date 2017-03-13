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
        uint32_t input_h = -1, input_w = -1, input_c = -1;
        vector<string> layer_names;
        map<string, DM_Layer_Param *> layer_names_to_layer_params;
        bool failed_to_read = false;
    public:
        DM_Net_Parameter(string net_dir_path) {
            string main_file_path = net_dir_path + "/main.dm";

            ifstream in(main_file_path.c_str());
            Json::Value net;
            in >> net;

            this->num_layers = net["NUM_LAYERS"].asUInt();
            this->input_c = net["INPUT_C"].asUInt();
            this->input_w = net["INPUT_W"].asUInt();
            this->input_h = net["INPUT_H"].asUInt();
            this->use_dm_layout = net["USE_DM_LAYOUT"].asBool();

            for (Json::Value::iterator it = net["LAYERS"].begin(); it != net["LAYERS"].end(); ++it) {
                string name((*it)["name"].asString());
                string type((*it)["type"].asString());
                string conf_path((*it)["conf_file"].asString());
                string w_path((*it)["weights_file"].asString());

                /*
                 * FixMe: Please check correct type
                 */

                layer_names.push_back(name);

                DM_Layer_Param * layer_param = new DM_Layer_Param(name, type, net_dir_path, conf_path, w_path);

                pair<string, DM_Layer_Param*> pair(name, layer_param);
                layer_names_to_layer_params.insert(pair);
            }

            if(input_c <= 0 || input_h <= 0 || input_w <= 0)
                failed_to_read = true;

            if(num_layers != layer_names.size())
                failed_to_read = true;
        }
        uint32_t GetNumLayers() {
            return this->num_layers;
        }
        uint32_t GetInputH() {
            return this->input_h;
        }
        uint32_t GetInputW() {
            return this->input_w;
        }
        uint32_t GetInputC() {
            return this->input_c;
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
                LOGD("\tInput: (%d, %d, %d)", input_c, input_h, input_w);
                LOGD("\tNumber of layers: %d", num_layers);
                for (int i = 0; i < num_layers; i++) {
                    layer_names_to_layer_params.find(layer_names.at(i))->second->PrintLayerParam();
                }
            }
        }
    };
}

#endif
