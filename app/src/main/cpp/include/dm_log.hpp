/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   dm_log.hpp
 * Author: JC1DA
 *
 * Created on March 1, 2017, 4:43 PM
 */

#ifndef DM_LOG_HPP
#define DM_LOG_HPP

#include <stdio.h>
#include <android/log.h>

#define  LOG_TAG    "DEEPMON"
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

//LINUX
//#define LOGD(...) printf(__VA_ARGS__); printf("\n")
//#define LOGE(...) printf(__VA_ARGS__); printf("\n")


#endif /* DM_LOG_HPP */

