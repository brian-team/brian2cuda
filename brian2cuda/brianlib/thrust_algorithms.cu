#include "brianlib/cuda_utils.h"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include <stdint.h>

namespace brian {

struct is_in_subgroup
{
    int32_t start;
    int32_t stop;

    __host__ __device__
    bool operator()(const int32_t& neuron) const
    {
        return start <= neuron && neuron < stop;
    }
};

int filter_subgroup_eventspace(
        int32_t* src, int n, int32_t* dst, int32_t start, int32_t stop)
{
    if (n <= 0)
        return 0;

    thrust::device_ptr<int32_t> src_ptr(src);
    thrust::device_ptr<int32_t> dst_ptr(dst);
    is_in_subgroup pred{start, stop};

    thrust::device_ptr<int32_t> end = dst_ptr;
    THRUST_CHECK_ERROR(
        end = thrust::copy_if(src_ptr, src_ptr + n, dst_ptr, pred)
    );
    return static_cast<int>(end - dst_ptr);
}

}  // namespace brian
