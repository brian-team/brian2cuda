#include "brianlib/device_buffer.h"
#include "brianlib/cuda_utils.h"

#include <thrust/device_vector.h>

namespace brian {

template<typename T>
struct DeviceBuffer<T>::Impl
{
    thrust::device_vector<T> vec;
};

template<typename T>
DeviceBuffer<T>::DeviceBuffer()
    : impl_(std::make_unique<Impl>()), ptr_(nullptr), n_(0)
{
}

template<typename T>
DeviceBuffer<T>::~DeviceBuffer() = default;

template<typename T>
DeviceBuffer<T>::DeviceBuffer(DeviceBuffer&& other) noexcept
    : impl_(std::move(other.impl_)), ptr_(other.ptr_), n_(other.n_)
{
    other.ptr_ = nullptr;
    other.n_ = 0;
}

template<typename T>
DeviceBuffer<T>& DeviceBuffer<T>::operator=(DeviceBuffer&& other) noexcept
{
    if (this != &other)
    {
        impl_ = std::move(other.impl_);
        ptr_ = other.ptr_;
        n_ = other.n_;
        other.ptr_ = nullptr;
        other.n_ = 0;
    }
    return *this;
}

template<typename T>
void DeviceBuffer<T>::refresh_ptr()
{
    if (!impl_ || impl_->vec.empty())
        ptr_ = nullptr;
    else
        ptr_ = thrust::raw_pointer_cast(impl_->vec.data());
}

template<typename T>
void DeviceBuffer<T>::resize(size_t n)
{
    if (!impl_)
        impl_ = std::make_unique<Impl>();
    if (n == 0)
    {
        clear();
        return;
    }
    THRUST_CHECK_ERROR(impl_->vec.resize(n));
    n_ = n;
    refresh_ptr();
}

template<typename T>
void DeviceBuffer<T>::copy_from_host(const T* src, size_t n)
{
    resize(n);
    if (n == 0 || src == nullptr)
        return;
    CUDA_SAFE_CALL(cudaMemcpy(
        ptr_, src, n * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void DeviceBuffer<T>::copy_to_host(T* dst) const
{
    if (n_ == 0 || dst == nullptr || ptr_ == nullptr)
        return;
    CUDA_SAFE_CALL(cudaMemcpy(
        dst, ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void DeviceBuffer<T>::store(size_t i, const T& elem)
{
    if (i >= n_ || ptr_ == nullptr)
        return;
    CUDA_SAFE_CALL(cudaMemcpy(
        ptr_ + i, &elem, sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void DeviceBuffer<T>::append(const T& elem)
{
    const size_t i = n_;
    resize(n_ + 1);
    store(i, elem);
}

template<typename T>
void DeviceBuffer<T>::clear()
{
    if (impl_)
    {
        THRUST_CHECK_ERROR(impl_->vec.clear());
        THRUST_CHECK_ERROR(impl_->vec.shrink_to_fit());
    }
    ptr_ = nullptr;
    n_ = 0;
}

// Explicit instantiations for types used by generated brian2cuda code.
template class DeviceBuffer<char>;
template class DeviceBuffer<int32_t>;
template class DeviceBuffer<int64_t>;
template class DeviceBuffer<uint32_t>;
template class DeviceBuffer<uint64_t>;
template class DeviceBuffer<float>;
template class DeviceBuffer<double>;
template class DeviceBuffer<char*>;
template class DeviceBuffer<int32_t*>;
template class DeviceBuffer<int64_t*>;
template class DeviceBuffer<uint32_t*>;
template class DeviceBuffer<uint64_t*>;
template class DeviceBuffer<float*>;
template class DeviceBuffer<double*>;

}  // namespace brian
