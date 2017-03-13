#ifndef DM_LAYER_PARAM_HPP
#define DM_LAYER_PARAM_HPP

namespace deepmon {
    class DM_Layer_Param {
    private:
        std::string name;
        std::string conf_path;
        std::string weights_path;
    public:

        explicit DM_Layer_Param(std::string name, std::string conf_path, std::string weights_path) {
            this->name = name;
            this->conf_path = conf_path;
            this->weights_path = weights_path;
        }

        std::string get_name() {
            return this->name;
        }
        std::string get_conf_path() {
            return this->conf_path;
        }
        std::string get_weight_path() {
            return this->weights_path;
        };
    };
}

#endif