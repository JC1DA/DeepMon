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

#include <layers/dm_layer_fc.hpp>
#include <fstream>
#include <json/json.h>
#include <cblas.h>

namespace deepmon {
    DM_Layer_Fc::DM_Layer_Fc(DM_Layer_Param &param) : DM_Layer(param.GetName(), param.GetType(), param.GetInputLayersNames(), param.GetMemoryLayout()) {
        //read config file
        ifstream in(param.GetConfPath().c_str());
        Json::Value layer;
        in >> layer;

        //save weights path
        this->weights_path = param.GetWeightsPath();

        this->has_bias = layer["HAS_BIAS"].asBool();

        this->env = (layer["USE_GPU"].asBool()) ? ENVIRONMENT_GPU : ENVIRONMENT_CPU;
        if(this->env == ENVIRONMENT_GPU)
            this->precision = (layer["USE_HALF"].asBool()) ? PRECISION_16 : PRECISION_32;

        this->num_neurons = layer["NUM_NEURONS"].asUInt();

        if(!weights_path.compare("") || this->num_neurons < 1) {
            this->corrupted = true;
            return;
        }
    }

    void DM_Layer_Fc::ComputeOutputShapes(vector<vector<uint32_t >> inputs_shapes_no_batches) {
        if(inputs_shapes_no_batches.size() != 1) {
            LOGE("Invalid Input's Shapes");
            this->corrupted = true;
            return;
        }

        vector<uint32_t> input_shapes = inputs_shapes_no_batches.at(0);

        //store this shape for loading weights
        this->inputs_shapes.push_back(input_shapes);

        input_size = 1;
        for(int i = 0 ; i < input_shapes.size() ; i++) {
            input_size *= input_shapes.at(i);
        }

        this->filters_shapes.push_back(num_neurons);
        this->filters_shapes.push_back(input_size);

        this->output_shapes.push_back(num_neurons);
    }

    void DM_Layer_Fc::LoadWeights() {
        float *bias_data = NULL;
        float *weights_data = NULL;

        if(1) {
            FILE *fp = fopen(this->weights_path.c_str(), "r");
            if(this->has_bias) {
                bias_data = new float[this->num_neurons];
                fread((void*)bias_data, sizeof(float), this->num_neurons, fp);
                this->biases = new DM_Blob(vector<uint32_t>{this->num_neurons}, this->env, this->precision, bias_data);
                delete bias_data;
            }

            weights_data = new float[filters_shapes.at(0) * filters_shapes.at(1)];
            if(mem_layout == MEMORY_LAYOUT_CAFFE)
                fread((void*)weights_data, sizeof(float), filters_shapes.at(0) * filters_shapes.at(1), fp);
            else {
                /*
                 * Fixme: has to look at prev layer to know the shapes of input
                 */
                vector<uint32_t> prev_layer_shapes = this->inputs_shapes.at(0);
                if(prev_layer_shapes.size() == 1) {
                    /*
                     * FIXME: should take consideration of input memory layout
                     * default: [input_size x output_size] - need to transpose
                     */

                    float *buffer = (float *) calloc(num_neurons, sizeof(float));
                    for(int i = 0 ; i < input_size ; i++) {
                        fread(buffer, sizeof(float), num_neurons, fp);
                        for(int n = 0 ; n < num_neurons ; n++) {
                            weights_data[n * input_size + i] = buffer[n];
                        }
                    }
                    free(buffer);
                } else if(prev_layer_shapes.size() == 3) {

                    /*
                     * Input memory layout: [input_size x output_size]
                     * input_size = [c x h x w] if prev layer is conv layer
                     */

                    //prev layer is conv layer
                    int input_c = prev_layer_shapes.at(2);
                    int input_h = prev_layer_shapes.at(0);
                    int input_w = prev_layer_shapes.at(1);

                    float *buffer = (float *) calloc(num_neurons, sizeof(float));
                    for(int c = 0 ; c < input_c ; c++) {
                        for(int h = 0 ; h < input_h ; h++) {
                            for(int w = 0 ; w < input_w ; w++) {
                                fread(buffer, sizeof(float), num_neurons, fp);
                                for(int n = 0 ; n < num_neurons ; n++) {
                                    int idx = ((n * input_h + h) * input_w + w) * input_c + c;
                                    weights_data[idx] = buffer[n];
                                }
                            }
                        }
                    }
                    free(buffer);
                }
            }
            fclose(fp);

            this->filters = new DM_Blob(this->filters_shapes, this->env, this->precision, weights_data);
            delete weights_data;
        } else {
            if(this->has_bias) {
                bias_data = new float[this->num_neurons];
                for(int i = 0 ; i < this->num_neurons ; i++)
                    bias_data[i] = 1.0f;
                this->biases = new DM_Blob(vector<uint32_t>{this->num_neurons}, this->env, this->precision, bias_data);
                delete bias_data;
            }

            weights_data = new float[this->input_size * this->num_neurons];

            for(int i = 0 ; i < input_size * num_neurons ; i++)
                weights_data[i] = 1;

            this->filters = new DM_Blob(filters_shapes, this->env, this->precision, weights_data);
            delete weights_data;
        }
    }
}