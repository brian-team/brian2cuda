#include "brianlib/device_buffer.h"
#include "brianlib/cuda_utils.h"

#include <thrust/device_vector.h>

#include <cstdio>
#include <cstdlib>

namespace brian {

struct DeviceBuffer::Impl
{
    thrust::device_vector<char> bytes;
};

DeviceBuffer::DeviceBuffer()
    : impl_(std::make_unique<Impl>()), ptr_(nullptr), elem_size_(0), n_(0)
{
}

DeviceBuffer::DeviceBuffer(size_t elem_size)
    : impl_(std::make_unique<Impl>()), ptr_(nullptr), elem_size_(elem_size), n_(0)
{
}

DeviceBuffer::~DeviceBuffer() = default;

void DeviceBuffer::refresh_ptr()
{
    if (impl_->bytes.empty())
        ptr_ = nullptr;
    else
        ptr_ = thrust::raw_pointer_cast(impl_->bytes.data());
}

void DeviceBuffer::resize(size_t n)
{
    if (elem_size_ == 0)
    {
        fprintf(stderr, "ERROR: DeviceBuffer used before elem_size was set\n");
        exit(EXIT_FAILURE);
    }
    if (n == 0)
    {
        clear();
        return;
    }
    THRUST_CHECK_ERROR(impl_->bytes.resize(n * elem_size_));
    n_ = n;
    refresh_ptr();
}

void DeviceBuffer::copy_from_host(const void* src, size_t n)
{
    resize(n);
    if (n == 0 || src == nullptr)
        return;
    CUDA_SAFE_CALL(cudaMemcpy(
        ptr_, src, n * elem_size_, cudaMemcpyHostToDevice));
}

void DeviceBuffer::copy_to_host(void* dst) const
{
    if (n_ == 0 || dst == nullptr || ptr_ == nullptr)
        return;
    CUDA_SAFE_CALL(cudaMemcpy(
        dst, ptr_, n_ * elem_size_, cudaMemcpyDeviceToHost));
}

void DeviceBuffer::clear()
{
    THRUST_CHECK_ERROR(impl_->bytes.clear());
    THRUST_CHECK_ERROR(impl_->bytes.shrink_to_fit());
    ptr_ = nullptr;
    n_ = 0;
}

}  // namespace brian
