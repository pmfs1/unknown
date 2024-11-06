#ifndef __UNKNOWN__
#define __UNKNOWN__

#include "cortex.h"
#include "population.h"
#include "utils.h"

#ifdef __CUDACC__
#include "unknown_cuda.h"
#else
#include "unknown_std.h"
#endif

#endif
