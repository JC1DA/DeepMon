#include <jni.h>
#include <string>
#include <dm.hpp>
#include <dm_net.hpp>
#include <clblast_c.h>

using namespace deepmon;

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

    //need to test clblast
    const size_t m = 128;
    const size_t n = 64;
    const size_t k = 512;

    float* host_a = (float*)malloc(sizeof(float)*m*k);
    float* host_b = (float*)malloc(sizeof(float)*n*k);
    float* host_c = (float*)malloc(sizeof(float)*m*n);
    for (size_t i=0; i<m*k; ++i) { host_a[i] = 12.193f; }
    for (size_t i=0; i<n*k; ++i) { host_b[i] = -8.199f; }
    for (size_t i=0; i<m*n; ++i) { host_c[i] = 0.0f; }

    cl_context context = dm.GetGpuExecutionEngine().GetContext();
    cl_command_queue queue = dm.GetGpuExecutionEngine().GetCurrentQueue();

    cl_mem device_a = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, m*k*sizeof(float), host_a, NULL);
    cl_mem device_b = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, n*k*sizeof(float), host_b, NULL);
    cl_mem device_c = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, m*n*sizeof(float), host_c, NULL);

    const float alpha = 0.7f;
    const float beta = 1.0f;
    const size_t a_ld = k;
    const size_t b_ld = n;
    const size_t c_ld = n;

    cl_event event = NULL;
    CLBlastStatusCode status = CLBlastSgemm(CLBlastLayoutRowMajor,
                                            CLBlastTransposeNo, CLBlastTransposeNo,
                                            m, n, k,
                                            alpha,
                                            device_a, 0, a_ld,
                                            device_b, 0, b_ld,
                                            beta,
                                            device_c, 0, c_ld,
                                            &queue, &event);

    // Wait for completion
    if (status == CLBlastSuccess) {
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
    }

    // Example completed. See "clblast_c.h" for status codes (0 -> success).
    LOGD("Completed SGEMM with status %d\n", status);
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
    net->PrintProcessingPileline();

    int size = 2 * 3 * 3 * 3;
    float *data = new float[size];

    //Caffe
    for(int i = 0 ; i < size ; i++)
        data[i] = 1;


    /*int idx = 0;
    for(int b = 0 ; b < 2 ; b++) {
        for(int h = 0 ; h < 3 ; h++) {
            for(int w = 0 ; w < 3 ; w++) {
                int d = h * 3 + w;
                for(int c = 0 ; c < 2 ; c++) {
                    data[idx] = d;
                    idx++;
                }
            }
        }
    }*/

    DM_Blob *input = new DM_Blob(vector<uint32_t>{2,3,3,3}, ENVIRONMENT_CPU, PRECISION_32, data);
    free(data);

    net->Forward(input);
}
