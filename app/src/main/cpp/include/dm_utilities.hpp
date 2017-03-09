#ifndef DM_UTILITIES_HPP
#define DM_UTILITIES_HPP

#include "dm.hpp"

namespace deepmon {
	void test_im2col(DeepMon &dm);
    void test_openblas();
	void print_blob_3d(DM_Blob &blob, int dimension);
}

#endif