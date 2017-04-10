
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

        if(env == ENVIRONMENT_CPU) {
            if(0) {
                FILE *fp = fopen(this->weights_path.c_str(), "r");
                if(this->has_bias) {
                    bias_data = new float[this->num_neurons];
                    fread((void*)bias_data, sizeof(float), this->num_neurons, fp);
                    this->biases = new DM_Blob(vector<uint32_t>(this->num_neurons), this->env, this->precision, bias_data);
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
                        //normal read
                        fread((void*)weights_data, sizeof(float), filters_shapes.at(0) * filters_shapes.at(1), fp);
                    } else if(prev_layer_shapes.size() == 3) {
                        //prev layer is conv layer
                        for(int i = 0 ; i < this->num_neurons ; i++) {
                            for(int j = 0 ; j < prev_layer_shapes.at(2) ; j++) { //c
                                for(int m = 0 ; m < prev_layer_shapes.at(1) ; m++) {  //h
                                    for(int n = 0 ; n < prev_layer_shapes.at(0) ; n++) { //w
                                        int new_idx = ((i * prev_layer_shapes.at(2) + j) * prev_layer_shapes.at(1) + m) * prev_layer_shapes.at(0) + n;
                                        fread((void *)(&weights_data[new_idx]), sizeof(float), 1, fp);
                                    }
                                }
                            }
                        }
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
                    weights_data[i] = 1.0f;

                this->filters = new DM_Blob(filters_shapes, this->env, this->precision, weights_data);
                delete weights_data;
            }
        }
    }

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

        DM_Blob *result = NULL;

        int batches = input->get_shape_at(0);

        result = new DM_Blob(vector<uint32_t> {
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

    DM_Blob* DM_Layer_Fc::ForwardGpu(vector<DM_Blob *> blobs) {
        return NULL;
    }
}