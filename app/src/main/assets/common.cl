#ifndef PRECISION
  #define PRECISION 32      // Data-types: half, single or double precision, complex or regular
#endif

#if PRECISION == 16
  #pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif

// Half-precision
#if PRECISION == 16
  typedef half real;
  typedef half2 real2;
  typedef half4 real4;
  typedef half8 real8;
  typedef half16 real16;
  #define ZERO 0
  #define ONE 1
  #define SMALLEST -1.0e14

// Single-precision
#elif PRECISION == 32
  typedef float real;
  typedef float2 real2;
  typedef float4 real4;
  typedef float8 real8;
  typedef float16 real16;
  #define ZERO 0.0f
  #define ONE 1.0f
  #define SMALLEST -1.0e37f
#endif

#ifndef VWM
    #define VWM 4
#endif

#ifdef VWM
    #if VWM == 1
        typedef real realM;
        #define vloadM(X) vload(0, &(X));
        #define dotM(X,Y) ((X) * (Y))
    #elif VWM == 2
        typedef real2 realM;
        #define vloadM(X) vload2(0, &(X));
        #define dotM(X,Y) dot((X),(Y))
    #elif VWM == 4
        typedef real4 realM;
        #define vloadM(X) vload4(0, &(X));
        #define dotM(X,Y) dot((X),(Y))
    #elif VWM == 8
        typedef real8 realM;
        #define vloadM(X) vload8(0, &(X));
        #define dotM(X,Y) (dot((X.s0123),(Y.s0123)) + dot((X.s4567),(Y.s4567)))
    #elif VWM == 16
        typedef real16 realM;
        #define vloadM(X) vload16(0, &(X));
        #define dotM(X,Y) (dot((X.s0123),(Y.s0123)) + dot((X.s4567),(Y.s4567)) + dot((X.s89ab),(Y.s89ab)) + dot((X.scdef),(Y.scdef)))
    #endif
#endif

#ifndef USE_CL_MAD
  #define USE_CL_MAD 0
#endif

static inline int getIndexFrom3D(int d1, int d2, int d3, int i1, int i2, int i3) {
	return i1 * (d2 * d3) + i2 * d3 + i3;
}

static inline int getIndexFrom4D(int d1, int d2, int d3, int d4, int i1, int i2, int i3, int i4) {
	return i1 * (d2 * d3 * d4) + i2 * (d3 * d4) + i3 * d4 + i4;
}

#if PRECISION == 16
__kernel void convertFloatToHalf(
    __global const float *input,
    __global half *output) {
    int idx = get_global_id(0);
    vstore_half(input[idx], 0, &output[idx]);
}

__kernel void convertHalfToFloat(
    __global const half *input,
    __global float *output) {
    int idx = get_global_id(0);
    output[idx] = convert_float(input[idx]);
    //output[idx] = (float)input[idx];
}
#endif
