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
    deepmon::DeepMon dm = deepmon::DeepMon::Get();
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

    deepmon::DeepMon dm = deepmon::DeepMon::Get(package_path);
    std::vector<int> shapes({1,2,3});
    deepmon::DM_Blob *blob = NULL;
    blob = new deepmon::DM_Blob(shapes, deepmon::ENVIRONMENT_CPU, deepmon::PRECISION_32, NULL);
    blob = new deepmon::DM_Blob(shapes, deepmon::ENVIRONMENT_GPU, deepmon::PRECISION_32, NULL);
    blob = new deepmon::DM_Blob(shapes, deepmon::ENVIRONMENT_GPU, deepmon::PRECISION_16, NULL);
    float *data = new float[1 * 2 * 3];
    blob = new deepmon::DM_Blob(shapes, deepmon::ENVIRONMENT_CPU, deepmon::PRECISION_32, data);
    blob = new deepmon::DM_Blob(shapes, deepmon::ENVIRONMENT_GPU, deepmon::PRECISION_32, data);
    blob = new deepmon::DM_Blob(shapes, deepmon::ENVIRONMENT_GPU, deepmon::PRECISION_16, data);
}