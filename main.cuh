#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "types.cuh"
#include "visualize.cuh"
#include "random_utils.cuh"
#include "initialize.cuh"
#include "ColorUtils.cuh"
#include "fitness.cuh"
#include "tournament.cuh"

