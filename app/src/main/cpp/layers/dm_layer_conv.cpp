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

#include <layers/dm_layer_conv.hpp>
#include <json/json.h>
#include <fstream>
#include <dm_layer_param.hpp>
#include <dm_layer.hpp>
#include <cblas.h>

using namespace deepmon;

namespace deepmon {
    DM_Layer_Conv::DM_Layer_Conv(DM_Layer_Param &param) : DM_Layer(param.GetName(), param.GetType(), param.GetInputLayersNames(), param.GetMemoryLayout()) {
        //read config file
        ifstream in(param.GetConfPath().c_str());
        Json::Value layer;
        in >> layer;

        //save weights path
        this->weights_path = param.GetWeightsPath();

        this->num_filters = layer["NUM_FILTERS"].asUInt();
        this->num_channels = layer["NUM_CHANNELS"].asUInt();
        this->filter_h = layer["FILTER_H"].asUInt();
        this->filter_w = layer["FILTER_W"].asUInt();
        this->has_bias = layer["HAS_BIAS"].asBool();

        this->env = (layer["USE_GPU"].asBool()) ? ENVIRONMENT_GPU : ENVIRONMENT_CPU;
        if(this->env == ENVIRONMENT_GPU)
            this->precision = (layer["USE_HALF"].asBool()) ? PRECISION_16 : PRECISION_32;

        this->pad_left = layer["PAD_LEFT"].asUInt();
        this->pad_right = layer["PAD_RIGHT"].asUInt();
        this->pad_top = layer["PAD_TOP"].asUInt();
        this->pad_bottom = layer["PAD_BOTTOM"].asUInt();

        this->stride_h = layer["STRIDE_H"].asUInt();
        this->stride_w = layer["STRIDE_W"].asUInt();

        this->dilation_h = layer["DILATION_H"].asUInt();
        this->dilation_w = layer["DILATION_W"].asUInt();

        if(num_filters <= 0 || num_channels <= 0 || filter_h <= 0 || filter_w <= 0 ) {
            corrupted = true;
            return;
        }

        switch (param.GetMemoryLayout()) {
            case MEMORY_LAYOUT_DM:
                this->filters_shapes.push_back(num_filters);
                this->filters_shapes.push_back(filter_h);
                this->filters_shapes.push_back(filter_w);
                this->filters_shapes.push_back(num_channels);
                break;
            case MEMORY_LAYOUT_CAFFE:
                this->filters_shapes.push_back(num_filters);
                this->filters_shapes.push_back(num_channels);
                this->filters_shapes.push_back(filter_h);
                this->filters_shapes.push_back(filter_w);
                break;
            default:
                LOGE("Invalid Memory Layout");
        }
    }

    void DM_Layer_Conv::LoadWeights() {
        float *bias_data = NULL;
        float *weights_data = NULL;

        if(1) {
            /*
             * There are no differences between loading weights into CPU or GPU memory
             * Fixme: Memory consumption might be very large in this step
             */
            FILE *fp = fopen(this->weights_path.c_str(), "r");
            if(this->has_bias) {
                bias_data = new float[this->num_filters];
                fread((void*)bias_data, sizeof(float), this->num_filters, fp);
                this->biases = new DM_Blob(vector<uint32_t>{this->num_filters}, this->env, this->precision, bias_data);
                delete bias_data;
            }
            weights_data = new float[this->num_filters * this->num_channels * this->filter_h * this->filter_w];
            if(this->mem_layout == MEMORY_LAYOUT_CAFFE)
                fread((void*)weights_data, sizeof(float), this->num_filters * this->num_channels * this->filter_h * this->filter_w, fp);
            else if(this->mem_layout == MEMORY_LAYOUT_DM) {
                //convert Caffe-based weights into DM-based weights
                for(int i = 0 ; i < this->num_filters ; i++) {
                    for(int j = 0 ; j < this->num_channels ; j++) {
                        for(int m = 0 ; m < this->filter_h ; m++) {
                            for(int n = 0 ; n < this->filter_w ; n++) {
                                int new_idx = ((i * filter_h + m) * filter_w + n) * num_channels + j;
                                fread((void *)(&weights_data[new_idx]), sizeof(float), 1, fp);
                            }
                        }
                    }
                }
            }
            this->filters = new DM_Blob(filters_shapes, this->env, this->precision, weights_data);
            delete weights_data;
            fclose(fp);
        } else {
            if(this->has_bias) {
                bias_data = new float[this->num_filters];
                for(int i = 0 ; i < this->num_filters ; i++)
                    bias_data[i] = 1.0f;
                this->biases = new DM_Blob(vector<uint32_t>{this->num_filters}, this->env, this->precision, bias_data);
                delete bias_data;
            }

            weights_data = new float[this->num_filters * this->num_channels * this->filter_h * this->filter_w];
            for(int i = 0 ; i < num_filters * num_channels * filter_h * filter_w ; i++)
                weights_data[i] = 1.0f;
            this->filters = new DM_Blob(filters_shapes, this->env, this->precision, weights_data);
            delete weights_data;
        }
    }

    void DM_Layer_Conv::ComputeOutputShapes(vector<vector<uint32_t >> inputs_shapes_no_batches) {
        if(inputs_shapes_no_batches.size() != 1) {
            LOGE("Invalid Input's Shapes");
            this->corrupted = true;
            return;
        }

        vector<uint32_t> input_shapes = inputs_shapes_no_batches.at(0);
        uint32_t input_channels;
        if(mem_layout == MEMORY_LAYOUT_CAFFE) {
            input_channels = input_shapes.at(0);
            input_h = input_shapes.at(1);
            input_w = input_shapes.at(2);
        }
        else if(mem_layout == MEMORY_LAYOUT_DM) {
            input_channels = input_shapes.at(2);
            input_h = input_shapes.at(0);
            input_w = input_shapes.at(1);
        }

        if(this->num_channels != input_channels) {
            LOGE("%s: Incorrect number of channels (%d != %d)", this->name.c_str(), this->num_channels, input_channels);
            this->corrupted = true;
            return;
        }

        this->output_h = (input_h + pad_top + pad_bottom - (dilation_h * (filter_h - 1) + 1)) / stride_h + 1;
        this->output_w = (input_w + pad_left + pad_right - (dilation_w * (filter_w - 1) + 1)) / stride_w + 1;

        if(output_h <= 0 || output_w <= 0) {
            LOGE("%s: Incorrect output size (%d, %d)", name.c_str(), output_h, output_w);
            this->corrupted = true;
            return;
        }

        if(mem_layout == MEMORY_LAYOUT_DM) {
            this->output_shapes.push_back(output_h);
            this->output_shapes.push_back(output_w);
            this->output_shapes.push_back(num_filters);
        } else if(mem_layout == MEMORY_LAYOUT_CAFFE){
            this->output_shapes.push_back(num_filters);
            this->output_shapes.push_back(output_h);
            this->output_shapes.push_back(output_w);
        }
    }

    DM_Blob* DM_Layer_Conv::ForwardCpu(vector<DM_Blob *> blobs) {
        if(blobs.size() != 1) {
            LOGE("[%s] has more than 1 input", this->name.c_str());
            return NULL;
        }

        return this->do_conv_cpu(blobs[0]);
    }

    DM_Blob* DM_Layer_Conv::ForwardGpu(vector<DM_Blob *> blobs) {
        if(blobs.size() != 1) {
            LOGE("[%s] has more than 1 input", this->name.c_str());
            return NULL;
        }

        DM_Blob *input = blobs[0];

        if(input == NULL || input->get_shapes().size() != 4) {
            LOGE("[%s]: Invalid Number of Dims (%d != 4) !!!", name.c_str(), input->get_shapes().size());
            return NULL;
        }

        if(this->mem_layout == MEMORY_LAYOUT_CAFFE && input->get_shape_at(CAFFE_BLOB_INOUT_CHANNELS_IDX) != num_channels) {
            LOGE("[%s]: Incorrect number of channels (%d != %d)", name.c_str(), input->get_shape_at(CAFFE_BLOB_INOUT_CHANNELS_IDX), num_channels);
            return NULL;
        }

        return this->do_conv_gpu(input);
    }
}