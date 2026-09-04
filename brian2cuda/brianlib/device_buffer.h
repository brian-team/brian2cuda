#ifndef BRIAN_DEVICE_BUFFER_H
#define BRIAN_DEVICE_BUFFER_H

#include <cstddef>
#include <memory>

namespace brian {

// Type-erased resizable device storage. Header is Thrust-free;
// thrust::device_vector<char> lives only in device_buffer.cu behind Impl (PImpl).
// One TU instantiates a single device_vector type instead of one per dtype.
class DeviceBuffer
{
public:
    DeviceBuffer();
    explicit DeviceBuffer(size_t elem_size);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }

    void* data() { return ptr_; }
    const void* data() const { return ptr_; }

    template<typename T>
    T* data_as() { return static_cast<T*>(data()); }
    template<typename T>
    const T* data_as() const { return static_cast<const T*>(data()); }

    void set_elem_size(size_t elem_size) { elem_size_ = elem_size; }

    void resize(size_t n);
    void copy_from_host(const void* src, size_t n);
    void copy_to_host(void* dst) const;
    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    // Cached from impl_->bytes; must stay in sync via refresh_ptr().
    void* ptr_;
    size_t elem_size_;
    size_t n_;

    void refresh_ptr();
};

}  // namespace brian

#endif
