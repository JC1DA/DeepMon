#include <jni.h>
#include <string>
#include <dm.hpp>

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
    DeepMon *dm = new DeepMon();
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

    DeepMon *dm = new DeepMon(package_path);
}