/* 
Implementation taken of https://github.com/ThunderStruct/Color-Utilities/blob/master/ColorUtils.hpp
modified to work with CUDA
*/

#ifndef ColorUtils_hpp
#define ColorUtils_hpp

#include <cuda_runtime.h>

class ColorUtils
{
public:
    struct xyzColor
    {
        float x, y, z;
        
        __host__ __device__ xyzColor(){}
        __host__ __device__ xyzColor(float x, float y, float z) : x(x), y(y), z(z) {}
    };

    struct CIELABColorSpace
    {
        float l, a, b;
        
        __host__ __device__ CIELABColorSpace(){}
        __host__ __device__ CIELABColorSpace(float l, float a, float b) : l(l), a(a), b(b) {}
    };

    struct rgbColor
    {
        unsigned int r, g, b;
        
        __host__ __device__ rgbColor(){}
        __host__ __device__ rgbColor(unsigned int r, unsigned int g, unsigned int b) : r(r % 256), g(g % 256), b(b % 256) {}
    };
};

#endif /* ColorUtils_hpp */