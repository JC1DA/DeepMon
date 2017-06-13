//
// Created by JC1DA on 3/13/17.
//


#include <string>
#include <dm_net.hpp>
#include <dm.hpp>
#include <layers/dm_layer_conv.hpp>
#include <layers/dm_layer_data.hpp>
#include <layers/dm_layer_pooling.hpp>
#include <layers/dm_layer_softmax.hpp>
#include <layers/dm_layer_fc.hpp>
#include <layers/dm_layer_relu.hpp>
#include <layers/dm_layer_activation.hpp>

using namespace std;
using namespace deepmon;
namespace deepmon {
    DM_Net::DM_Net(string model_dir_path) {
        DM_Net_Parameter *net_param = new DM_Net_Parameter(model_dir_path);
        if(net_param->IsCorrupted()) {
            this->is_working = false;
            return;
        }

        for(int i = 0 ; i < net_param->GetLayerNames().size() ; i++) {
            string layer_name = net_param->GetLayerNames().at(i);
            DM_Layer_Param param = net_param->GetLayerParam(layer_name);
            LOGD("Parsing layer %s", layer_name.c_str());
            DM_Layer *layer = NULL;

            if(!param.GetType().compare(LAYER_NAME_DATA)) {
                layer = new DM_Layer_Data(param);
            } else if(!param.GetType().compare(LAYER_NAME_CONV)) {
                layer = new DM_Layer_Conv(param);
            } else if(!param.GetType().compare(LAYER_NAME_POOLING)) {
                layer = new DM_Layer_Pooling(param);
            } else if(!param.GetType().compare(LAYER_NAME_FULLY_CONNECTED)) {
                layer = new DM_Layer_Fc(param);
            } else if(!param.GetType().compare(LAYER_NAME_SOFTMAX)) {
                layer = new DM_Layer_Softmax(param);
            } else if(!param.GetType().compare(LAYER_NAME_ACTIVATION)) {
                layer = new DM_Layer_Activation(param);
            }
            layers.push_back(layer);

            pair<string, DM_Layer *> pair(layer_name, layer);
            name_to_layer_map.insert(pair);
        }

        /*
         * Fixme: Why the fuck I could not delete this object
         */
        //delete net_param;

        //create processing chains
        this->pipeline.push_back(layers.at(0));
        vector<string> processed_layers({layers.at(0)->GetName()});
        vector<string> pending_layers;

        for(int i = 1 ; i < layers.size() ; i++) {
            pending_layers.push_back(layers.at(i)->GetName());
        }

        bool has_cycle = false;
        while(!has_cycle && pending_layers.size() != 0) {
            DM_Layer *chosen_layer = NULL;
            int chosen_idx = -1;
            for(int i = 0 ; i < pending_layers.size() ; i++) {
                string layer_name = pending_layers.at(i);
                DM_Layer *layer = name_to_layer_map.find(layer_name)->second;
                vector<string> bottom_layers = layer->GetBottomLayersNames();

                bool all_bottoms_availabe = true;
                for(int j = 0 ; j < bottom_layers.size() ; j++) {
                    bool found_layer = false;
                    for(int k = 0 ; k < processed_layers.size() ; k++) {
                        if(bottom_layers.at(j) == processed_layers.at(k)) {
                            found_layer = true;
                            break;
                        }
                    }
                    if(!found_layer) {
                        all_bottoms_availabe = false;
                        break;
                    }
                }

                if(all_bottoms_availabe) {
                    chosen_layer = layer;
                    chosen_idx = i;
                    break;
                }
            }

            if(chosen_layer == NULL) {
                has_cycle = true;
                break;
            }

            pipeline.push_back(chosen_layer);
            processed_layers.push_back(chosen_layer->GetName());
            pending_layers.erase(pending_layers.begin() + chosen_idx);
        }

        if(has_cycle) {
            LOGE("Network has cycle");
            this->is_working = false;
            return;
        }

        //update tops_layers
        for(int i = pipeline.size() - 1 ; i >= 0 ; i--) {
            DM_Layer *layer = pipeline.at(i);
            for(int j = 0 ; j < layer->GetBottomLayersNames().size() ; j++) {
                DM_Layer *bottom_layer = name_to_layer_map.find(layer->GetBottomLayersNames().at(j))->second;
                bottom_layer->AppendTopLayer(layer->GetName());
            }
        }

        //create output-shapes and check
        for(int i = 0 ; i < pipeline.size() ; i++) {
            if(i == 0) {
                pipeline.at(i)->ComputeOutputShapes(vector<vector<uint32_t>>());
            } else {
                vector<vector<uint32_t>> prev_inputs;
                vector<string> prev_layers_names = pipeline.at(i)->GetBottomLayersNames();
                for(int j = 0 ; j < prev_layers_names.size() ; j++) {
                    prev_inputs.push_back(name_to_layer_map.find(prev_layers_names.at(j))->second->GetOutputShapes());
                }
                pipeline.at(i)->ComputeOutputShapes(prev_inputs);

                if(pipeline.at(i)->IsCorrupted()) {
                    LOGE("%s has been corrupted", pipeline.at(i)->GetName().c_str());
                    return;
                }
            }
        }

        //load weights - implement later
        for(int i = 0 ; i < pipeline.size() ; i++) {
            pipeline.at(i)->LoadWeights();
        }
    }

    DM_Blob * DM_Net::Forward(DM_Blob *input_blob) {
        if(!IsWorking()) {
            return NULL;
        }

        //push input_blob into data layer
        this->pipeline.at(0)->EnqueueInputBlob(input_blob);
        DM_Blob *result = NULL;

        for(int i = 0 ; i < pipeline.size() ; i++) {
            LOGD("Processing layer %s", pipeline.at(i)->GetName().c_str());

            result = pipeline.at(i)->Forward();

            if(result == NULL || result->is_corrupted()) {
                break;
            }

            DM_Blob *tmp = result->ConvertToCpuBlob();
            delete result;
            result = tmp;

            //send to upper layer's queues
            vector<string> top_layers_names = pipeline.at(i)->GetTopLayersNames();
            for(int j = 0 ; j < top_layers_names.size() ; j++) {
                name_to_layer_map.find(top_layers_names.at(j))->second->EnqueueInputBlob(result);
            }

            /*if(i == 1) {
                DM_Blob *tmp = result->ConvertToCpuBlob();
                tmp->print_blob();
                delete tmp;
            }*/
        }

        if(result != NULL && result->is_corrupted()) {
            delete result;
            result = NULL;
        }

        if(result != NULL) {
            //process final blob
            DM_Blob *final_result = result->ConvertToCpuBlob();

            //free result if needed
            if(!pipeline.at(pipeline.size() - 1)->IsUsingPersistentBlob()) {
                delete result;
            }

            result = final_result;
            //result->print_blob();
        }


        return result;
    }
}