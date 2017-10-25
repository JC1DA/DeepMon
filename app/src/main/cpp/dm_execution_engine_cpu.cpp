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

#include <dm_execution_engine_cpu.hpp>
#include <dm_blob.hpp>

namespace deepmon {
    DM_Execution_Engine_CPU::DM_Execution_Engine_CPU() : DM_Execution_Engine(ENVIRONMENT_CPU) {
        this->initialized = true;
    }

    void DM_Execution_Engine_CPU::AllocateMemory(DM_Blob *blob, float *initialized_data) {
        if(blob->get_env() == this->evn) {
            uint32_t size_in_bytes = blob->get_size() * sizeof(float);
            blob->set_mem_size(size_in_bytes);

            float *data = new float[size_in_bytes / sizeof(float)];
            if(initialized_data != NULL)
                memcpy(data, initialized_data, size_in_bytes);
            blob->set_cpu_data(data);
        } else {
            blob->set_corrupted(true);
        }
    }
}