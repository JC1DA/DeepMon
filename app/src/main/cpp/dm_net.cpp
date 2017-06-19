/*The MIT License (MIT)
 *
 *Copyright (c) 2013 Thomas Park
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *       of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *       to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *       copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *       The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *THE SOFTWARE.
 */


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
#include <cstdlib>

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

        //for debugging
        int grouth_truth_idx_mapping = 0;
        int checking_gt_failed = 0;

        for(int i = 0 ; i < pipeline.size() ; i++) {
            LOGD("Processing layer %s", pipeline.at(i)->GetName().c_str());

            result = pipeline.at(i)->Forward();

            if(result == NULL || result->is_corrupted()) {
                break;
            }

            //debugging gpu
            if(0) {
                DM_Blob *tmp = result->ConvertToCpuBlob();
                delete result;
                result = tmp;

                //debugging - compare output to grouth-truth
                if(!checking_gt_failed) {
                    if(i > 0 && i < pipeline.size() - 1) {
                        if(pipeline.at(i + 1)->GetType().compare(LAYER_NAME_ACTIVATION)) {
                            char gt_file_path[512]; gt_file_path[0] = '\0';
                            sprintf(gt_file_path,"/sdcard/dump/l_%d",grouth_truth_idx_mapping);

                            float *gt_data = (float *)malloc(result->get_size() * sizeof(float));

                            FILE *fp = fopen(gt_file_path, "r");
                            fread(gt_data, result->get_size(), sizeof(float), fp);
                            fclose(fp);

                            //compare with gt data
                            for(int z = 0 ; z < result->get_size() ; z++) {
                                float distance = gt_data[z] - result->get_cpu_data()[z];
                                float delta = 0.006;
                                if(-delta < distance && distance < delta) {
                                    continue;
                                } else {
                                    LOGD("result[%d] != gt[%d] (%f != %f)",z, z, result->get_cpu_data()[z], gt_data[z]);
                                    checking_gt_failed = 1;
                                    break;
                                }
                            }

                            free(gt_data);

                            grouth_truth_idx_mapping++;
                        }
                    }
                }
            }

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