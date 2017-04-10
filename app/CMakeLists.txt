# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

set(distribution_DIR ${CMAKE_SOURCE_DIR}/../distribution)
set(source_DIR ${CMAKE_SOURCE_DIR}/src/main/cpp)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library(lib_opencl SHARED IMPORTED)
set_target_properties( lib_opencl PROPERTIES IMPORTED_LOCATION ${distribution_DIR}/opencl/lib/${ANDROID_ABI}/libOpenCL.so )
add_library(lib_jsoncpp SHARED IMPORTED)
set_target_properties( lib_jsoncpp PROPERTIES IMPORTED_LOCATION ${distribution_DIR}/jsoncpp/lib/${ANDROID_ABI}/libjsoncpp.so )
add_library(lib_clblast STATIC IMPORTED)
set_target_properties( lib_clblast PROPERTIES IMPORTED_LOCATION ${distribution_DIR}/clblast/lib/${ANDROID_ABI}/libclblast.a )
add_library(lib_openblas STATIC IMPORTED)
set_target_properties( lib_openblas PROPERTIES IMPORTED_LOCATION ${distribution_DIR}/openblas/lib/${ANDROID_ABI}/libopenblas.a )

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -mfpu=vfpv3-d16 -march=armv7-a -fPIE -fPIC -fvisibility=default")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mfpu=vfpv3-d16 -march=armv7-a -fPIE -fPIC -fvisibility=default")
#set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}    -Xlinker --no-warn-mismatch -O2 -mfpu=vfpv3-d16 -mhard-float -D_NDK_MATH_NO_SOFTFP=1 -march=armv7-a -mfloat-abi=hard")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Xlinker --no-warn-mismatch -O2 -mfpu=vfpv3-d16 -mhard-float -D_NDK_MATH_NO_SOFTFP=1 -march=armv7-a -mfloat-abi=hard -std=gnu++11")

add_library( # Sets the name of the library.
             deepmon

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             src/main/cpp/native-lib.cpp
             ${source_DIR}/dm.cpp
             ${source_DIR}/dm_execution_engine_cpu.cpp
             ${source_DIR}/dm_execution_engine_gpu.cpp
             ${source_DIR}/dm_net.cpp
             ${source_DIR}/dm_blob.cpp
             ${source_DIR}/layers/dm_layer_conv.cpp
             ${source_DIR}/layers/dm_layer_conv_cpu.cpp
             ${source_DIR}/layers/dm_layer_conv_gpu.cpp
             ${source_DIR}/layers/dm_layer_data.cpp
             ${source_DIR}/layers/dm_layer_pooling.cpp
             ${source_DIR}/layers/dm_layer_pooling_cpu.cpp
             ${source_DIR}/layers/dm_layer_pooling_gpu.cpp
             ${source_DIR}/layers/dm_layer_softmax.cpp
             ${source_DIR}/layers/dm_layer_fc.cpp)

target_include_directories(deepmon PRIVATE
                                ${distribution_DIR}/opencl/include
                                ${distribution_DIR}/openblas/include
                                ${distribution_DIR}/jsoncpp/include
                                ${distribution_DIR}/clblast/include)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/include)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       deepmon

                       lib_opencl
                       lib_openblas
                       lib_jsoncpp
                       lib_clblast

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib} )