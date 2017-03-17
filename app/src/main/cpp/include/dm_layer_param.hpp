#ifndef DM_LAYER_PARAM_HPP
#define DM_LAYER_PARAM_HPP

using namespace std;

namespace deepmon {
    class DM_Layer_Param {
    private:
        MEMORY_LAYOUT layout;
        string name;
        string type;
        string model_dir_path;
        string conf_path;
        string weights_path;
        vector<string> inputs;
        bool persistent_blobs = false;
    public:

        explicit DM_Layer_Param(string name, string type, string model_dir_path, string conf_path, string weights_path) {
            this->name = name;
            this->type = type;
            this->model_dir_path = model_dir_path;
            this->conf_path = conf_path;
            this->weights_path = weights_path;
            this->layout = MEMORY_LAYOUT_DM;
        }

        explicit DM_Layer_Param(string name, string type, string model_dir_path, string conf_path, string weights_path, bool use_dm_layout) {
            this->name = name;
            this->type = type;
            this->model_dir_path = model_dir_path;
            this->conf_path = conf_path;
            this->weights_path = weights_path;
            this->layout = use_dm_layout ? MEMORY_LAYOUT_DM : MEMORY_LAYOUT_CAFFE;
        }

        explicit DM_Layer_Param(string name, string type, string model_dir_path, string conf_path, string weights_path, vector<string> inputs, bool use_dm_layout) {
            this->name = name;
            this->type = type;
            this->model_dir_path = model_dir_path;
            this->conf_path = conf_path;
            this->weights_path = weights_path;
            this->inputs = inputs;
            this->layout = use_dm_layout ? MEMORY_LAYOUT_DM : MEMORY_LAYOUT_CAFFE;
        }

        explicit DM_Layer_Param(string name, string type, string model_dir_path, \
                                string conf_path, string weights_path, vector<string> inputs, \
                                bool use_dm_layout, bool use_persistent_blobs) {
            this->name = name;
            this->type = type;
            this->model_dir_path = model_dir_path;
            this->conf_path = conf_path;
            this->weights_path = weights_path;
            this->inputs = inputs;
            this->layout = use_dm_layout ? MEMORY_LAYOUT_DM : MEMORY_LAYOUT_CAFFE;
            this->persistent_blobs = use_persistent_blobs;
        }

        string GetName() {
            return this->name;
        }
        string GetType() {
            return this->type;
        }
        string GetConfPath() {
            return model_dir_path + "/" + conf_path;
        }
        string GetWeightsPath() {
            return model_dir_path + "/" + weights_path;
        };
        MEMORY_LAYOUT GetMemoryLayout() {
            return layout;
        }
        vector<string> GetInputLayersNames() {
            return vector<string>(inputs);
        }
        bool IsUsingPersistentBlobs() {
            return persistent_blobs;
        }
        void PrintLayerParam() {
            LOGD("Layer's Name: %s", name.c_str());
            LOGD("\tTYPE: %s", type.c_str());
            LOGD("\tCONF_FILE: %s", conf_path.c_str());
            LOGD("\tWEIGHTS_FILE: %s", weights_path.c_str());
        }
    };
}

#endif