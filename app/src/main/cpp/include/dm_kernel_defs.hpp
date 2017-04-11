#ifndef DM_KERNEL_DEFS
#define DM_KERNEL_DEFS

namespace deepmon {
#define KERNEL_CONVERT_FLOAT_TO_HALF    "convertFloatToHalf"
#define KERNEL_CONVERT_HALF_TO_FLOAT    "convertHalfToFloat"
#define KERNEL_MEMCPY                   "memcpy"
#define KERNEL_CAFFE_IM2COL             "caffe_im2col"
#define KERNEL_CAFFE_COL2IM             "caffe_col2im"
#define KERNEL_DM_CONV_BASE             "dm_conv_base"
#define KERNEL_CAFFE_MAXPOOL            "caffe_maxpool"
#define KERNEL_CAFFE_AVEPOOL            "caffe_avepool"

//Activation functions
#define KERNEL_ACTIVATE_RELU            "activate_relu"
#define KERNEL_ACTIVATE_TANH            "activate_tanh"
#define KERNEL_ACTIVATE_SIGMOID         "activate_sigmoid"
}

#endif
