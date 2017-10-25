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

        DM_Blob *Forward(DM_Blob *blob);
        DM_Blob *ForwardCache(DM_Blob *blob);

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
                /*for(int i = 0 ; i < this->layers.size() ; i++)
                    this->layers.at(i)->PrintInfo();*/
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

        vector<uint32_t> GetInputShapes() {
            vector<uint32_t> shapes;
            shapes.push_back(1);
            vector<uint32_t> input_shapes = pipeline.at(0)->GetOutputShapes();
            for(int i = 0 ; i < input_shapes.size() ; i++)
                shapes.push_back(input_shapes.at(i));
            return shapes;
        }

        vector<uint32_t> GetOutputShapes() {

            /*
             * FIXME: some models have multiple outputs
             * One approach to fix is to identify in configuration file
             * Store temp output only needed to save memory
             */

            vector<uint32_t> shapes;
            shapes.push_back(1);
            vector<uint32_t> last_layer_shapes = pipeline.at(pipeline.size() - 1)->GetOutputShapes();
            for(int i = 0 ; i < last_layer_shapes.size() ; i++)
                shapes.push_back(last_layer_shapes.at(i));
            return shapes;
        }

        uint32_t GetOutputSize() {
            uint32_t size = 1;
            vector<uint32_t> output_shapes = GetOutputShapes();
            for(int i = 0 ; i < output_shapes.size() ; i++)
                size *= output_shapes.at(i);
            return size;
        }

        void SetUpCaching(int total_non_cached_blocks, int *non_cached_indices_x, int *non_cached_indices_y) {
            for(int i = 0 ; i < pipeline.size() ; i++) {
                if(pipeline.at(i)->IsCachable()) {
                    pipeline.at(i)->SetUpCaching(total_non_cached_blocks, non_cached_indices_x, non_cached_indices_y);
                }
            }
        }
    };
}

#endif
