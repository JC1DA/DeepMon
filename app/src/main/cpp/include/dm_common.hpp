/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   dm_common.hpp
 * Author: JC1DA
 *
 * Created on March 1, 2017, 8:07 PM
 */

#ifndef DM_COMMON_HPP
#define DM_COMMON_HPP

#include <string.h>

namespace deepmon {
    typedef enum {
        ENVIRONMENT_CPU,
        ENVIRONMENT_GPU
    } ENVIRONMENT_TYPE;

    typedef enum {
        PRECISION_32,
        PRECISION_16
    } PRESICION_TYPE;

    typedef enum {
        MEMORY_LAYOUT_DM,
        MEMORY_LAYOUT_CAFFE
    } MEMORY_LAYOUT;

    typedef enum {
        CAFFE_BLOB_INOUT_BATCH_IDX,
        CAFFE_BLOB_INOUT_CHANNELS_IDX,
        CAFFE_BLOB_INOUT_HEIGHT_IDX,
        CAFFE_BLOB_INOUT_WIDTH_IDX
    } CAFFE_BLOB_INOUT_IDX;

    typedef enum {
        DM_BLOB_INOUT_BATCH_IDX,
        DM_BLOB_INOUT_HEIGHT_IDX,
        DM_BLOB_INOUT_WIDTH_IDX,
        DM_BLOB_INOUT_CHANNELS_IDX
    } DM_BLOB_INOUT_IDX;

    typedef enum {
        CAFFE_BLOB_FILTER_NUM_FILTERS,
        CAFFE_BLOB_FILTER_NUM_CHANNELS,
        CAFFE_BLOB_FILTER_HEIGHT,
        CAFFE_BLOB_FILTER_WIDTH
    } CAFFE_BLOB_FILTER_IDX;

    typedef enum {
        DM_BLOB_FILTER_NUM_FILTERS,
        DM_BLOB_FILTER_HEIGHT,
        DM_BLOB_FILTER_WIDTH,
        DM_BLOB_FILTER_NUM_CHANNELS,
    } DM_BLOB_FILTER_IDX;

#define INPUT_NAME      "DATA"
#define CONV_NAME       "CONV"
#define POOLING_NAME    "POOLING"
#define FC_NAME         "FULLY_CONNECTED"
#define SOFTMAX_NAME    "SOFTMAX"

    inline bool CMP_OPTION(char *str, const char *option) {
        bool ret = strncmp(str, option, strlen(option)) == 0 ? true : false;
        return ret;
    }
}

#endif /* DM_COMMON_HPP */

