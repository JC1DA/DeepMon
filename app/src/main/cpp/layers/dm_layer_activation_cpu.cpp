//
// Created by JC1DA on 6/7/17.
//

#include <layers/dm_layer_activation.hpp>

namespace deepmon {
    void DM_Layer_Activation::Activation_ReLU_CPU(DM_Blob *input, DM_Blob *output) {
        float *data_in = input->get_cpu_data();
        float *data_out = output->get_cpu_data();
        for(int i = 0 ; i < output->get_total_size() ; i++)
            data_out[i] = data_in[i] > 0 ? data_in[i] : 0;
    }

    void DM_Layer_Activation::Activation_Leaky_CPU(DM_Blob *input, DM_Blob *output) {
        float *data_in = input->get_cpu_data();
        float *data_out = output->get_cpu_data();
        for(int i = 0 ; i < output->get_total_size() ; i++)
            data_out[i] = data_in[i] > -this->activation_threshold ? data_in[i] : 0;
    }
}
