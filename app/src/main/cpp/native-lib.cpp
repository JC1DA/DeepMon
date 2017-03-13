#include <jni.h>
#include <string>
#include <dm.hpp>
#include <dm_log.hpp>
#include "dm_utilities.hpp"
#include "dm_net.hpp"

using namespace deepmon;

extern "C"
JNIEXPORT jstring JNICALL
Java_com_lanytek_deepmon_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_lanytek_deepmon_MainActivity_testDeepMon(
        JNIEnv* env,
        jobject /* this */) {
    DeepMon dm = DeepMon::Get();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_lanytek_deepmon_MainActivity_testDeepMonWithPackageName(
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

    //test_im2col(dm);
    //test_openblas();
    //test_conv_cpu();
}

extern "C"
JNIEXPORT void JNICALL
        Java_com_lanytek_deepmon_MainActivity_testLoadNet(
                JNIEnv* env,
                jobject thisobj/* this */,
                jstring model_dir_path) {
    const char *model_dir_path_str = env->GetStringUTFChars(model_dir_path, 0);
    std::string path(model_dir_path_str);
    env->ReleaseStringUTFChars(model_dir_path, model_dir_path_str);

    DM_Net *net = new DM_Net(path);
    net->PrintNet();
}