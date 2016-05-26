#ifndef _CUDA_VECTOR_H_
#define _CUDA_VECTOR_H_

#include <cstdio>

/*
 * current memory allocation strategy:
 * only grow larger (new_size = old_size*2 + 1) ~= 2^n
 */

#define INITIAL_SIZE 1

template <class scalar>
class cudaVector
{
private:
	scalar* data;		//pointer to allocated memory
	int size_allocated;	//how much memory is allocated, should ALWAYS >= size_used
	int size_used;		//how many elements are stored in this vector

public:
	__device__ cudaVector()
	{
		size_used = 0;
		if(INITIAL_SIZE > 0)
		{
			data = (scalar*)malloc(sizeof(scalar) * INITIAL_SIZE);
			if(data)
			{
				size_allocated = INITIAL_SIZE;
			}
			else
			{
				printf("ERROR while creating vector with size %d in cudaVector.h/resize()\n", sizeof(scalar)*INITIAL_SIZE);
			}
		}
	};

	__device__ ~cudaVector()
	{
		free(data);
	};

	__device__ scalar* getDataPointer()
	{
		return data;
	};

	__device__ scalar at(int index)
	{
		if(index <= size_used && index >= 0)
		{
			return data[index];
		}
		else
		{
			return 0;
		}
	};

	__device__ void push(scalar elem)
	{
		if(size_allocated == size_used)
		{
			//resize larger
			resize(size_allocated*2 + 1);
		}
		if(size_used < size_allocated)
		{
			data[size_used] = elem;
			size_used++;
		}
	};

	__device__ void update(unsigned int pos, scalar elem)
	{
		if(pos <= size_used)
		{
			data[pos] = elem;
		}
		else
		{
			printf("ERROR invalid index %d, must be in range 0 - %d\n", pos, size_used);
		}
	};

	__device__ void resize(unsigned int new_capacity)
	{
		if(new_capacity > size_allocated)
		{
			//realloc larger memory (deviceside realloc doesn't exist, so we write our own)
			scalar* new_data = (scalar*)malloc(sizeof(scalar) * new_capacity);
			if (new_data)
			{
				memcpy(new_data, data, sizeof(scalar)*size());
				free(data);
				data = new_data;
				size_allocated = new_capacity;
			}
			else
			{
				printf("ERROR while resizing vector to size %d in cudaVector.h/resize()\n", sizeof(scalar)*new_capacity);
			}
		}
		else
		{
			//kleiner reallocen?
			size_used = new_capacity;
		};
	};

	//does not overwrite old data, just resets number of elements stored to 0
	__device__ void reset()
	{
		size_used = 0;
	};

	__device__ int size()
	{
		return size_used;
	};
};

#endif
