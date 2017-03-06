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
set_target_properties( lib_opencl PROPERTIES IMPORTED_LOCATION ${distribution_DIR}/opencl/lib/${ANDROID_ABI}/libGLES_mali.so )

add_library( # Sets the name of the library.
             deepmon

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             src/main/cpp/native-lib.cpp
             ${source_DIR}/dm.cpp
             ${source_DIR}/dm_err.cpp
             ${source_DIR}/dm_execution_engine.cpp
             ${source_DIR}/dm_execution_engine_gpu.cpp
             ${source_DIR}/dm_execution_engine_cpu.cpp
             ${source_DIR}/dm_kernel_object.cpp
             ${source_DIR}/dm_blob.cpp)

target_include_directories(deepmon PRIVATE ${distribution_DIR}/opencl/include)
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
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       deepmon

                       lib_opencl

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib} )