#include <celero/Celero.h>

#if defined(ALLOCATION_BENCHMARK) || defined(ALL)
#include \
    "allocation.cu"
#endif

#if defined(LOCATION_BENCHMARK) || defined(ALL)
#include \
    "location.cu"
#endif

CELERO_MAIN
