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

#include <layers/dm_layer_softmax.hpp>
#include <math.h>

namespace deepmon {
    DM_Layer_Softmax::DM_Layer_Softmax(DM_Layer_Param &param) : DM_Layer(param.GetName(), param.GetType(), param.GetInputLayersNames(), param.GetMemoryLayout()) {
        this->env = ENVIRONMENT_CPU;
        this->precision = PRECISION_32;
    }

    void DM_Layer_Softmax::ComputeOutputShapes(
            vector<vector<uint32_t >> inputs_shapes_no_batches) {
        if(inputs_shapes_no_batches.size() != 1) {
            LOGE("Invalid Input's Shapes");
            this->corrupted = true;
            return;
        }

        vector<uint32_t> input_shapes = inputs_shapes_no_batches.at(0);

        uint32_t output_size = 1;
        for(int i = 0 ; i < inputs_shapes_no_batches.size() ; i++)
            output_size *= input_shapes.at(i);

        this->output_shapes.push_back(output_size);
    }

    DM_Blob* DM_Layer_Softmax::ForwardCpu(vector<DM_Blob *> blobs) {

        if(blobs.size() != 1) {
            LOGE("[%s] has more than 1 input", this->name.c_str());
            return NULL;
        }

        DM_Blob *result = blobs[0];

        for(int b = 0 ; b < result->get_shape_at(0) ; b++) {
            float *output = result->get_cpu_data() + b * this->output_shapes.at(0);

            float dsum = 0;
            for(int i = 0 ; i < this->output_shapes.at(0) ; i++) {
                dsum += exp((double)output[i]);
            }

            for(int i = 0 ; i < this->output_shapes.at(0) ; i++) {
                output[i] = (float)(exp((double)output[i]) / dsum);
            }
        }

        result->set_persistent(true);

        return result;
    }

    DM_Blob* DM_Layer_Softmax::ForwardGpu(vector<DM_Blob *> blobs) {
        return NULL;
    }
}