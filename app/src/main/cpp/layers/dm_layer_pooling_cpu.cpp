#include <layers/dm_layer_pooling.hpp>
#include <dm_layer_param.hpp>

namespace deepmon {
    void DM_Layer_Pooling::CAFFE_LAYOUT_ForwardCPU_MaxPool(DM_Blob *input, DM_Blob *output) {
        uint32_t batches = input->get_shape_at(0);

        float *bottom_data = input->get_cpu_data();
        float *top_data = output->get_cpu_data();

        for(int i = 0 ; i < output->get_total_size() ; i++)
            top_data[i] = -999999.999f;

        for(int b = 0 ; b < batches ; b++) {
            for(int c = 0 ; c < num_channels ; c++) {
                for(int ph = 0 ; ph < output_h ; ph++) {
                    for(int pw = 0 ; pw < output_w ; pw++) {
                        int hstart = ph * stride_h - pad_top;
                        int wstart = pw * stride_w - pad_left;
                        int hend = min(hstart + filter_h, input_h);
                        int wend = min(wstart + filter_w, input_w);
                        hstart = max(hstart, (int)0);
                        wstart = max(wstart, (int)0);

                        const int pool_index = ph * output_w + pw;
                        for (int h = hstart; h < hend; ++h) {
                            for (int w = wstart; w < wend; ++w) {
                                const int index = h * input_w + w;
                                if (bottom_data[index] > top_data[pool_index]) {
                                    top_data[pool_index] = bottom_data[index];
                                }
                            }
                        }
                    }
                }

                bottom_data += input_w * input_h;
                top_data += output_w * output_h;
            }
        }
    }

    void DM_Layer_Pooling::CAFFE_LAYOUT_ForwardCPU_AvePool(DM_Blob *input, DM_Blob *output) {
        uint32_t batches = input->get_shape_at(0);

        float *bottom_data = input->get_cpu_data();
        float *top_data = output->get_cpu_data();

        for(int i = 0 ; i < output->get_total_size() ; i++)
            top_data[i] = 0;

        for (int b = 0; b < batches; b++) {
            for (int c = 0; c < num_channels; ++c) {
                for (int ph = 0; ph < output_h; ++ph) {
                    for (int pw = 0; pw < output_w; ++pw) {
                        int hstart = ph * stride_h - pad_top;
                        int wstart = pw * stride_w - pad_left;
                        int hend = min(hstart + filter_h, input_h + pad_bottom);
                        int wend = min(wstart + filter_w, input_w + pad_right);
                        int pool_size = (hend - hstart) * (wend - wstart);
                        hstart = max(hstart, (int)0);
                        wstart = max(wstart, (int)0);
                        hend = min(hend, (int)input_h);
                        wend = min(wend, (int)input_w);
                        for (int h = hstart; h < hend; ++h) {
                            for (int w = wstart; w < wend; ++w) {
                                top_data[ph * output_w + pw] += bottom_data[h * input_w + w];
                            }
                        }
                        top_data[ph * output_w + pw] /= pool_size;
                    }
                }
                bottom_data += input_w * input_h;
                top_data += output_w * output_h;
            }
        }
    }

    void DM_Layer_Pooling::DM_LAYOUT_ForwardCPU_MaxPool(DM_Blob *input, DM_Blob *output) {
        uint32_t batches = input->get_shape_at(0);

        float *bottom_data = input->get_cpu_data();
        float *top_data = output->get_cpu_data();

        for(int i = 0 ; i < output->get_total_size() ; i++)
            top_data[i] = -999999.999f;

        for(int b = 0 ; b < batches ; b++) {
            for(int ph = 0 ; ph < output_h ; ph++) {
                for(int pw = 0 ; pw < output_w ; pw++) {
                    int hstart = ph * stride_h - pad_top;
                    int wstart = pw * stride_w - pad_left;

                    int base_idx = (ph * output_w + pw) * num_channels;

                    for(int y = 0 ; y < filter_h ; y++) {
                        int y_ = hstart + y;
                        for(int x = 0 ; x < filter_w ; x++) {
                            int x_ = wstart + x;
                            for(int c = 0 ; c < num_channels ; c++) {
                                float d = 0;

                                if(x_ < 0 || y_ < 0 || x_ >= input_w || y_ >= input_h)
                                    d = 0;
                                else
                                    d = bottom_data[(y_ * input_w + x_) * num_channels + c];

                                top_data[base_idx + c] = max(top_data[base_idx + c], d);
                            }
                        }
                    }
                }
            }
            bottom_data += num_channels * input_w * input_h;
            top_data += num_channels * output_w * output_h;
        }
    }

    void DM_Layer_Pooling::DM_LAYOUT_ForwardCPU_AvePool(DM_Blob *input, DM_Blob *output) {
        uint32_t batches = input->get_shape_at(0);

        float *bottom_data = input->get_cpu_data();
        float *top_data = output->get_cpu_data();

        for(int i = 0 ; i < output->get_total_size() ; i++)
            top_data[i] = 0;

        for(int b = 0 ; b < batches ; b++) {
            for(int ph = 0 ; ph < output_h ; ph++) {
                for(int pw = 0 ; pw < output_w ; pw++) {
                    int hstart = ph * stride_h - pad_top;
                    int wstart = pw * stride_w - pad_left;

                    int hend = min(hstart + filter_h, input_h + pad_bottom);
                    int wend = min(wstart + filter_w, input_w + pad_right);

                    int pool_size = (hend - hstart) * (wend - wstart);

                    int base_idx = (ph * output_w + pw) * num_channels;
                    for(int y = 0 ; y < filter_h ; y++) {
                        int y_ = hstart + y;
                        for(int x = 0 ; x < filter_w ; x++) {
                            int x_ = wstart + x;
                            for(int c = 0 ; c < num_channels ; c++) {
                                float d = 0;

                                if(x_ < 0 || y_ < 0 || x_ >= input_w || y_ >= input_h)
                                    d = 0;
                                else
                                    d = bottom_data[(y_ * input_w + x_) * num_channels + c];

                                top_data[base_idx + c] += d;
                            }
                        }
                    }

                    for(int c = 0 ; c < num_channels ; c++)
                        top_data[base_idx + c] /= pool_size;
                }
            }
            bottom_data += num_channels * input_w * input_h;
            top_data += num_channels * output_w * output_h;
        }
    }

    DM_Blob* DM_Layer_Pooling::do_pooling_cpu(DM_Blob *input) {

        DM_Blob *output = new DM_Blob(vector<uint32_t> {
                input->get_shape_at(0), output_shapes[0], output_shapes[1], output_shapes[2]
        }, ENVIRONMENT_CPU, PRECISION_32, NULL);

        if(mem_layout == MEMORY_LAYOUT_CAFFE) {
            if(!type.compare("MAXPOOL")) {
                CAFFE_LAYOUT_ForwardCPU_MaxPool(input, output);
            } else if(!type.compare("AVEPOOL")) {
                CAFFE_LAYOUT_ForwardCPU_AvePool(input, output);
            } else {
                LOGE("[%s] Incorrect Memory Pooling Type", this->name.c_str());
                output->set_corrupted(true);
            }
        } else if(mem_layout == MEMORY_LAYOUT_DM) {
            if(!type.compare("MAXPOOL")) {
                DM_LAYOUT_ForwardCPU_MaxPool(input, output);
            } else if(!type.compare("AVEPOOL")) {
                DM_LAYOUT_ForwardCPU_AvePool(input, output);
            } else {
                LOGE("[%s] Incorrect Memory Pooling Type", this->name.c_str());
                output->set_corrupted(true);
            }
        } else {
            LOGE("[%s] Incorrect Memory Layout", this->name.c_str());
            output->set_corrupted(true);
        }

        return output;
    }
}