/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   dm_err.hpp
 * Author: JC1DA
 *
 * Created on March 1, 2017, 5:33 PM
 */

#ifndef DM_ERR_HPP
#define DM_ERR_HPP

#include <CL/cl.h>
#include "dm_log.hpp"

const char* opencl_error_to_str (cl_int error);

#define SHOW_ERROR_LOG(ERR) \
    LOGD                                                                              \
        (                                                                             \
            "OpenCL error with code %s happened in file %s at line %d. Exiting.\n",   \
            opencl_error_to_str(ERR), __FILE__, __LINE__                              \
        );                                                                            

#define SAMPLE_CHECK_ERRORS(ERR)                                                      \
    if(ERR != CL_SUCCESS)                                                             \
    {                                                                                 \
        SHOW_ERROR_LOG(ERR)                                                           \
    }

#define SAMPLE_CHECK_ERRORS_WITH_ERR_RETURN(ERR) \
    if(ERR != CL_SUCCESS)                                                             \
    {                                                                                 \
        SHOW_ERROR_LOG(ERR)                                                           \
        return ERR;                                                                   \
    }

#define SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(ERR) \
    if(ERR != CL_SUCCESS)                                                             \
    {                                                                                 \
        SHOW_ERROR_LOG(ERR)                                                           \
        return NULL;                                                                  \
    }

#define SAMPLE_CHECK_ERRORS_WITH_FALSE_RETURN(ERR) \
    if(ERR != CL_SUCCESS)                                                             \
    {                                                                                 \
        SHOW_ERROR_LOG(ERR)                                                           \
        return false;                                                                 \
    }

#endif /* DM_ERR_HPP */

