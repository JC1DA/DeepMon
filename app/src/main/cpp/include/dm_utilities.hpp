#ifndef DM_UTILITIES_HPP
#define DM_UTILITIES_HPP

#include "dm.hpp"

namespace deepmon {
	void test_im2col(DeepMon &dm);
    void test_openblas();
	void print_blob_3d(DM_Blob &blob, int dimension);
	void matrix_multiplication(float *A, int A_width_, int A_height_,
							   float *B, int B_width_, int B_height_,
							   float *AB, bool tA, bool tB, float beta);
    void test_conv_cpu();
}

#endif