#ifndef _CUDA_VECTOR_H_
#define _CUDA_VECTOR_H_

#include <cstdio>
#include <assert.h>

/*
 * current memory allocation strategy:
 * only grow larger (new_size = old_size*2 + 1) ~= 2^n
 */

#define INITIAL_SIZE 1

typedef int size_type;

template <class scalar>
class cudaVector
{
private:
    // TODO: consider using data of type char*, since it does not have a cunstructor
    scalar* volatile m_data;        //pointer to allocated memory
    volatile size_type m_capacity;  //how much memory is allocated, should ALWAYS >= size
    volatile size_type m_size;      //how many elements are stored in this vector

public:
    __device__ cudaVector()
    {
        m_size = 0;
        if(INITIAL_SIZE > 0)
        {
            m_data = (scalar*)malloc(sizeof(scalar) * INITIAL_SIZE);
            if(m_data != NULL)
            {
                m_capacity = INITIAL_SIZE;
            }
            else
            {
                printf("ERROR while creating cudaVector with size %ld in cudaVector.h (constructor)\n", sizeof(scalar)*INITIAL_SIZE);
                assert(m_data != NULL);
            }
        }
    };

    __device__ ~cudaVector()
    {
        free(m_data);
    };

    __device__ scalar* getDataPointer()
    {
        return m_data;
    };

    __device__ scalar& at(size_type index)
    {
        if (index < 0 || index >= m_size)
        {
            // TODO: check for proper exception throwing in cuda kernels
            printf("ERROR returning a reference to index %d in cudaVector::at() (size = %u)\n", index, m_size);
            assert(index < m_size);
        }
        return m_data[index];
    };

    __device__ void push(scalar elem)
    {
        assert(m_size <= m_capacity);
        if(m_capacity == m_size)
        {
            // increase capacity
            reserve(m_capacity*2 + 1);
        }
        if(m_size < m_capacity)
        {
            m_data[m_size] = elem;
            m_size++;
        }
    };

    __device__ void update(size_type pos, scalar elem)
    {
        if(pos <= m_size)
        {
            m_data[pos] = elem;
        }
        else
        {
            printf("ERROR invalid index %d, must be in range 0 - %d\n", pos, m_size);
            assert(pos <= m_size);
        }
    };

    __device__ void resize(size_type new_size)
    {
        if (new_size > m_capacity)
            reserve(new_size * 2);
        m_size = new_size;
    }

    __device__ size_type increaseSizeBy(size_type add_size)
    {
        size_type old_size = m_size;
        size_type new_size = old_size + add_size;
        if (new_size > m_capacity)
            reserve(new_size * 2);
        m_size = new_size;
        return old_size;
    }

    __device__ void reserve(size_type new_capacity)
    {
        if(new_capacity > m_capacity)
        {
            //realloc larger memory (deviceside realloc doesn't exist, so we write our own)
            scalar* new_data = (scalar*)malloc(sizeof(scalar) * new_capacity);
            // TODO: use C++ version, is there a way to copy data in parallel here?
            //       since only num_unique_delays threads resize, the other threads could help copy?
            //scalar* new_data = new scalar[new_capacity];
            //if (new_data)
            //{
            //  for (size_type i = 0; i < m_size; i++)
            //      new_data[i] = m_data[i];
            //
            //  delete [] m_data;
            //  m_data = new_data;
            //  m_capacity = new_capacity;
            //}
            if (new_data != NULL)
            {
                memcpy(new_data, m_data, sizeof(scalar) * size());
                free(m_data);
                m_data = new_data;
                m_capacity = new_capacity;
            }
            else
            {
                printf("ERROR while allocating %ld bytes in cudaVector.h/reserve()\n", sizeof(scalar)*new_capacity);
                assert(new_data != NULL);
            }
        }
        else
        {
            //kleiner reallocen?
            m_capacity = new_capacity;
        };
    };

    //does not overwrite old data, just resets number of elements stored to 0
    __device__ void reset()
    {
        m_size = 0;
    };

    __device__ size_type size()
    {
        return m_size;
    };
};

#endif
