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
}

#endif /* DM_COMMON_HPP */

