#include <jni.h>
#include <string>
#include <dm.hpp>
#include <dm_net.hpp>
#include <clblast_c.h>
#include <cstdlib>

using namespace deepmon;

DM_Net *net = NULL;

extern "C"
JNIEXPORT jstring JNICALL
Java_com_lanytek_deepmon_MainActivity_stringFromJNI(
        JNIEnv *env,
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
Java_com_lanytek_deepmon_MainActivity_InitDeepMonWithPackageName(
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
Java_com_lanytek_deepmon_MainActivity_LoadNet(
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
        Java_com_lanytek_deepmon_MainActivity_GetInference(
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

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_lanytek_deepmon_MainActivity_TestInference(
        JNIEnv* env,
        jobject thisobj/* this */
) {
    float *data = (float *) malloc(3 * 448 * 448 * sizeof(float));
    FILE *fp = fopen("/sdcard/dump/input", "r");
    fread(data, 3 * 448 * 448, sizeof(float), fp);
    fclose(fp);

    DM_Blob *input = new DM_Blob(vector<uint32_t>{1, 448, 448, 3}, ENVIRONMENT_GPU, PRECISION_32, data);

    DM_Blob *result = net->Forward(input); //this is cpu blob

    jfloatArray resultArr = env->NewFloatArray(1470);
    env->SetFloatArrayRegion(resultArr, 0, 1470, result->get_cpu_data());

    return resultArr;
}
