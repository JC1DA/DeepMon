#ifndef DM_LAYER_HPP
#define DM_LAYER_HPP

#include <string>
#include <queue>
#include "dm_blob.hpp"
#include "dm_layer_param.hpp"
#include "dm_common.hpp"

using namespace std;

namespace deepmon {
	class DM_Layer {
	private:
	public:
		DM_Layer(string name, string type, vector<string> bottom_layers, MEMORY_LAYOUT mem_layout) {
            this->name = name;
            this->type = type;
            this->bottom_layers = bottom_layers;
            this->mem_layout = mem_layout;
        }
        bool IsCorrupted() {
            return corrupted;
        }
        bool IsUsingPersistentBlob() {
            return persistant_blobs;
        }
        string GetName() {
            return this->name;
        }
        vector<string> GetBottomLayersNames() {
            return vector<string>(bottom_layers);
        }
        vector<string> GetTopLayersNames() {
            return vector<string>(top_layers);
        }
        vector<uint32_t> GetOutputShapes() {
            return vector<uint32_t>(output_shapes);
        }
		virtual void LoadWeights() = 0;
        virtual void PrintInfo() = 0;
        virtual void ComputeOutputShapes(vector<vector<uint32_t >> inputs_shapes_no_batches) = 0;

        void AppendTopLayer(string layer_name) {
            top_layers.push_back(layer_name);
        }
        void EnqueueInputBlob(DM_Blob *input) {
            this->input_queue.push(input);
        }
        DM_Blob *Forward(DM_Blob *blob) {
            DM_Blob *result = NULL;
            if(env == ENVIRONMENT_CPU) {
                //call forward_cpu
            } else if(env == ENVIRONMENT_GPU) {
                //call forward_gpu
            } else {
                LOGE("Incorrect Environment");
                return NULL;
            }
            return result;
        }
	protected:
        string name;
        string type;
        bool persistant_blobs = false;
        bool corrupted = false;
        vector<string> bottom_layers;
        vector<string> top_layers;
		ENVIRONMENT_TYPE env = ENVIRONMENT_CPU;
        PRESICION_TYPE precision = PRECISION_32;
        MEMORY_LAYOUT mem_layout = MEMORY_LAYOUT_DM;
        vector<uint32_t> output_shapes;
        queue<DM_Blob *> input_queue;
	};
}

#endif