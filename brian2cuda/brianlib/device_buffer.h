#ifndef BRIAN_DEVICE_BUFFER_H
#define BRIAN_DEVICE_BUFFER_H

#include <cstddef>
#include <stdint.h>

namespace brian {

// Typed resizable device storage. Header is Thrust-free; thrust::device_vector<T>
// lives only in device_buffer.cu behind Impl (PImpl).
template<typename T>
class DeviceBuffer
{
public:
    DeviceBuffer();
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }

    void resize(size_t n);
    void copy_from_host(const T* src, size_t n);
    void copy_to_host(T* dst) const;
    void append(const T& elem);
    void store(size_t i, const T& elem);
    void clear();

private:
    struct Impl;
    Impl* impl_;
    T* ptr_;
    size_t n_;

    void refresh_ptr();
};

}  // namespace brian

#endif
