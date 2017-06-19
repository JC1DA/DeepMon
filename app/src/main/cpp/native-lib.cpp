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

#include <jni.h>
#include <string>
#include <dm.hpp>
#include <dm_net.hpp>
#include <clblast_c.h>
#include <cstdlib>

using namespace deepmon;

DM_Net *net = NULL;

extern "C"
JNIEXPORT void JNICALL
Java_com_lanytek_deepmon_DeepMon_InitDeepMonWithPackageName(
        JNIEnv* env,
        jobject thisobj/* this */,
        jstring package_name) {

    const char *packageNameStr = env->GetStringUTFChars(package_name, 0);
    std::string package_path("");
    package_path.append("/data/data/");
    package_path.append(packageNameStr);
    package_path.append("/app_execdir/");
    env->ReleaseStringUTFChars(package_name, packageNameStr);

    DeepMon dm = DeepMon::Get(package_path);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_lanytek_deepmon_DeepMon_LoadNet(
        JNIEnv* env,
        jobject thisobj/* this */,
        jstring model_dir_path) {
    const char *model_dir_path_str = env->GetStringUTFChars(model_dir_path, 0);
    std::string path(model_dir_path_str);
    env->ReleaseStringUTFChars(model_dir_path, model_dir_path_str);

    net = new DM_Net(path);
    net->PrintNet();
    net->PrintProcessingPileline();
}

extern "C"
JNIEXPORT jfloatArray JNICALL
        Java_com_lanytek_deepmon_DeepMon_GetInference(
                JNIEnv* env,
                jobject thisobj/* this */,
                jfloatArray input_arr
        ) {
    jfloat* data = env->GetFloatArrayElements(input_arr, 0);

    /*
     * Default only support 1 inference
     */

    DM_Blob *input = new DM_Blob(net->GetInputShapes(), ENVIRONMENT_GPU, PRECISION_32, data);
    env->ReleaseFloatArrayElements(input_arr, data, 0);

    DM_Blob *result = net->Forward(input); //this is cpu blob

    jfloatArray resultArr = env->NewFloatArray(net->GetOutputSize());
    env->SetFloatArrayRegion(resultArr, 0, net->GetOutputSize(), result->get_cpu_data());

    return resultArr;
}
