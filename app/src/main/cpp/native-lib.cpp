#include <jni.h>
#include <string>
#include <dm.hpp>
#include <dm_log.hpp>

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
    std::vector<int> shapes({3,4,5});

    float data[3*4*5];
    for(int i = 0 ; i < 3*4*5 ; i++)
        data[i] = i;

    //deepmon::DM_Blob *blob = new deepmon::DM_Blob(shapes, deepmon::ENVIRONMENT_CPU, deepmon::PRECISION_32, data);
    deepmon::DM_Blob *blob = new deepmon::DM_Blob(shapes, deepmon::ENVIRONMENT_GPU, deepmon::PRECISION_32, data);
    //deepmon::DM_Blob *blob = new deepmon::DM_Blob(shapes, deepmon::ENVIRONMENT_GPU, deepmon::PRECISION_32, data);

    //deepmon::DM_Blob *newblob = dm.get_excution_engine(false)->blob_convert_to_cpu_blob(blob);
    //deepmon::DM_Blob *newblob = dm.get_excution_engine(false)->blob_convert_to_gpu_blob(blob, deepmon::PRECISION_32);
    deepmon::DM_Blob *newblob = dm.get_excution_engine(false)->blob_convert_to_gpu_blob(blob, deepmon::PRECISION_16);


    deepmon::DM_Blob *cpu_blob = dm.get_excution_engine(false)->blob_convert_to_cpu_blob(newblob);
    for(int i = 0 ; i < 3*4*5 ; i++) {
        if(i != cpu_blob->get_cpu_data()[i]) {
            LOGD("Incorrect data at %d with data %f", i, cpu_blob->get_cpu_data()[i]);
            break;
        }
    }

    delete blob;
    delete newblob;
    delete cpu_blob;
}