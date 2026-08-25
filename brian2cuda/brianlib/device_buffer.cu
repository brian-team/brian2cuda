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

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : impl_(std::move(other.impl_)),
      ptr_(other.ptr_),
      elem_size_(other.elem_size_),
      n_(other.n_)
{
    other.ptr_ = nullptr;
    other.elem_size_ = 0;
    other.n_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept
{
    if (this != &other)
    {
        impl_ = std::move(other.impl_);
        ptr_ = other.ptr_;
        elem_size_ = other.elem_size_;
        n_ = other.n_;
        other.ptr_ = nullptr;
        other.elem_size_ = 0;
        other.n_ = 0;
    }
    return *this;
}

void DeviceBuffer::init(size_t elem_size)
{
    if (!impl_)
        impl_ = std::make_unique<Impl>();
    if (elem_size_ != 0 && elem_size_ != elem_size && n_ > 0)
        clear();
    elem_size_ = elem_size;
}

void DeviceBuffer::refresh_ptr()
{
    if (!impl_ || impl_->bytes.empty())
        ptr_ = nullptr;
    else
        ptr_ = thrust::raw_pointer_cast(impl_->bytes.data());
}

void DeviceBuffer::resize(size_t n)
{
    if (!impl_)
        impl_ = std::make_unique<Impl>();
    if (elem_size_ == 0)
    {
        fprintf(stderr, "ERROR: DeviceBuffer used before init()\n");
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

void DeviceBuffer::store(size_t i, const void* elem)
{
    if (elem == nullptr || i >= n_ || ptr_ == nullptr)
        return;
    CUDA_SAFE_CALL(cudaMemcpy(
        static_cast<char*>(ptr_) + i * elem_size_,
        elem,
        elem_size_,
        cudaMemcpyHostToDevice));
}

void DeviceBuffer::append(const void* elem)
{
    const size_t i = n_;
    resize(n_ + 1);
    store(i, elem);
}

void DeviceBuffer::clear()
{
    if (impl_)
    {
        THRUST_CHECK_ERROR(impl_->bytes.clear());
        THRUST_CHECK_ERROR(impl_->bytes.shrink_to_fit());
    }
    ptr_ = nullptr;
    n_ = 0;
}

}  // namespace brian
