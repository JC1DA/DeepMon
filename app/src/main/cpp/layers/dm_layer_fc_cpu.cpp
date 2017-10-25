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

#include <dm_blob.hpp>
#include <vector>
#include <dm_common.hpp>
#include <layers/dm_layer_fc.hpp>
#include <cblas.h>

using namespace std;
using namespace deepmon;

namespace deepmon {
    /*
     * Perform normal matrix multiplication because
     * input_blob always has dims B x N
     * output_blob always has dims O x N
     */
    DM_Blob* DM_Layer_Fc::ForwardCpu(vector<DM_Blob *> blobs) {

        if(blobs.size() != 1) {
            LOGE("[%s] has more than 1 input", this->name.c_str());
            return NULL;
        }

        DM_Blob *input = blobs[0];

        int batches = input->get_shape_at(0);

        DM_Blob *result = new DM_Blob(vector<uint32_t> {
                input->get_shape_at(0), output_shapes[0]
        }, ENVIRONMENT_CPU, PRECISION_32, NULL);

        float *data_in = input->get_cpu_data();
        float *data_out = result->get_cpu_data();

        int m = batches;
        int n = num_neurons;
        int k = input_size;

        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    m, n, k,
                    1.0f,
                    data_in, k,
                    filters->get_cpu_data(), k,
                    0, data_out, n);

        float *biases_multiplier = NULL;
        if(biases != NULL) {
            biases_multiplier = new float[n];
            for(int i = 0 ; i < n ; i++)
                biases_multiplier[i] = 1;
        }

        if(biases != NULL) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        batches, n, 1,
                        1.0f,
                        biases_multiplier, 1,
                        biases->get_cpu_data(), n,
                        1.0, data_out, n);
        }

        if(biases != NULL)
            delete biases_multiplier;

        return result;
    }
}