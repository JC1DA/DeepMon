#ifndef DM_LAYER_HPP
#define DM_LAYER_HPP

#include <string>
#include "dm_blob.hpp"
#include "dm_layer_param.hpp"

using namespace std;

namespace deepmon {
	class DM_Layer {
	private:
	public:
		DM_Layer(string name, string type, vector<string> inputs) {
            this->name = name;
            this->type = type;
            this->inputs = inputs;
        }
        bool IsCorrupted() {
            return corrupted;
        }
        string GetName() {
            return this->name;
        }
        vector<string> GetBottomLayersNames() {
            return vector<string>(inputs);
        }
		virtual void LoadWeights() = 0;
        virtual void PrintInfo() = 0;
	protected:
        DM_Layer_Param *param;
        string name;
        string type;
        vector<string> inputs;
		ENVIRONMENT_TYPE env = ENVIRONMENT_CPU;
        PRESICION_TYPE precision = PRECISION_32;
        bool corrupted = false;
	};
}

#endif