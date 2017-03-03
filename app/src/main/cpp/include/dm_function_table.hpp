/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   dm_function_table.hpp
 * Author: JC1DA
 *
 * Created on March 1, 2017, 8:27 PM
 */

#ifndef DM_FUNCTION_TABLE_HPP
#define DM_FUNCTION_TABLE_HPP

class DM_Function_Table {
public:
    virtual void do_conv(void *input, void *params, void *output);
};

#endif /* DM_FUNCTION_TABLE_HPP */

