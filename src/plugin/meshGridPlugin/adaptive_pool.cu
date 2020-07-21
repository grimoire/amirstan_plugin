#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <cuda_fp16.h>

#include "adaptive_pool.h"
#include "amir_cuda_util/cuda_util.h"

namespace amirstan
{
namespace plugin
{
    #define START_IND(a,b,c) (int)std::floor((float)(a * c) / b)
    #define END_IND(a,b,c) (int)std::ceil((float)((a + 1) * c) / b)

    using namespace amirstan::cuda;

    using amirstan::cuda::TensorSize;
    using amirstan::cuda::TensorStride;

    template <typename T>
    __host__ __device__ __forceinline__ T ceilDiv(T a, T b) {
        return (a + b - 1) / b;
    }


    template <typename T>
    struct idleOp {
            inline __device__ T operator()(const T& x, size_t reduce_count=0){
                return x;
            }
    };
    
    template <typename T>
    struct maxOp{
        inline __device__ T operator()(const T& x, const T& y){
            return x>y?x:y;
        }
    };
    
    template <typename T>
    struct sumOp{
        inline __device__ T operator()(const T& x, const T& y){
            return x+y;
        }
    };


    template <typename T>
    struct divCountOp {
            inline __device__ T operator()(const T& x, size_t reduce_count=0){
                return x/reduce_count;
            }
    };
    
    template<typename T, 
    typename PreReduceOp, 
    typename ReduceOp, 
    typename PostReduceOp>
    __global__ void adaptive_pool_kernel(T *output, const T* input,
                                        TensorSize input_size, TensorStride input_stride,
                                        TensorSize output_size, TensorStride output_stride,
                                        int nb_dims, int nb_reduce_dims, 
                                        int cell_per_block, int thread_per_cell,
                                        PreReduceOp pre_reduce_op,
                                        ReduceOp reduce_op,
                                        PostReduceOp post_reduce_op,
                                        size_t N)
                                        {

        const int REDUCE_MAX_COUNT=10;
        float reduce_size[REDUCE_MAX_COUNT];
        float reduce_stride[REDUCE_MAX_COUNT];
        
        extern __shared__ T smemChar[];

        for (int cell_id = blockIdx.x * cell_per_block + threadIdx.x/thread_per_cell; cell_id < (N); cell_id += cell_per_block * gridDim.x){
            // exit if thread belong to no cell
            if(threadIdx.x >= cell_per_block*thread_per_cell){
                break;
            }
            
            // how many element in cell and the start input offset of cell
            size_t cell_count = 1;
            size_t cell_input_offset = 0;
            int tmp_cell_id = cell_id;
            for(int i=0;i<nb_dims;++i){
                const int idx = tmp_cell_id/output_stride.size[i];
                tmp_cell_id = tmp_cell_id%output_stride.size[i];
                const int input_idx_start = START_IND(idx, output_size.size[i], input_size.size[i]);
                const int input_idx_end = END_IND(idx, output_size.size[i], input_size.size[i]);
                cell_count *= (input_idx_end-input_idx_start);
                reduce_size[i] = (input_idx_end-input_idx_start);
                cell_input_offset += input_idx_start*input_stride.size[i];
            }
            cell_count = max(cell_count, size_t(1));
            reduce_stride[nb_dims-1] = 1;
            for(int i=nb_dims-2; i>=0; --i){
                reduce_stride[i] = reduce_stride[i+1]*reduce_size[i+1];
            }

            // put data in share memory
            const int cell_thread_id = threadIdx.x%thread_per_cell;
            for(int cell_data_id=cell_thread_id; cell_data_id<cell_count; cell_data_id+=thread_per_cell){
                int tmp_cell_data_id = cell_data_id;
                size_t input_idx = 0;
                for(int i=nb_dims-nb_reduce_dims;i<nb_dims;++i){
                    const int idx = tmp_cell_data_id/reduce_stride[i];
                    tmp_cell_data_id = tmp_cell_data_id%int(reduce_stride[i]);

                    input_idx += idx*input_stride.size[i];
                }

                if(cell_data_id==cell_thread_id){
                    smemChar[threadIdx.x] = pre_reduce_op(input[cell_input_offset+input_idx], cell_count);
                }else{
                    smemChar[threadIdx.x] = reduce_op(smemChar[threadIdx.x], pre_reduce_op(input[cell_input_offset+input_idx], cell_count));
                }
            }
            __syncthreads();

            ///// reduce smemChar
            for(unsigned int reduce_step=thread_per_cell/2; reduce_step>0; reduce_step>>=1){
                if(cell_thread_id<reduce_step && cell_thread_id<cell_count){
                    smemChar[threadIdx.x] = reduce_op(smemChar[threadIdx.x], smemChar[threadIdx.x+reduce_step]);
                } 
                __syncthreads();
            }

            if(cell_thread_id==0){

                // for(int i=1;i<min(size_t(thread_per_cell),size_t(cell_count));++i){
                //     smemChar[threadIdx.x]=reduce_op(smemChar[threadIdx.x], smemChar[threadIdx.x+i]);
                // }
                output[cell_id] =  post_reduce_op(smemChar[threadIdx.x], cell_count);
            }

        }
    }


    template <typename T>
    void adaptive_pool(T *output, const T* input, 
                        int* input_dims, int* output_dims, int nb_dims,
                        int nb_reduce_dims,
                        PoolType pool_type,
                        cudaStream_t stream){

        TensorSize ts_input_size;
        TensorStride input_stride;

        memcpy(&(ts_input_size.size[0]), input_dims, sizeof(int)*nb_dims);
        input_stride.size[nb_dims-1] = 1;
        for(int i=nb_dims-2; i>=0; --i){
            input_stride.size[i] = input_stride.size[i+1] * ts_input_size.size[i+1];
        }

        TensorSize ts_output_size;
        TensorStride output_stride;
        memcpy(&ts_output_size.size[0], output_dims, sizeof(int)*nb_dims);
        output_stride.size[nb_dims-1] = 1;
        for(int i=nb_dims-2; i>=0; --i){
            output_stride.size[i] = output_stride.size[i+1] * ts_output_size.size[i+1];
        }

        size_t reduce_cell_count = 1;
        for(int i=nb_dims-nb_reduce_dims; i<nb_dims; ++i){
            size_t reduce_size = ceilDiv<size_t>(ts_input_size.size[i],ts_output_size.size[i]);
            reduce_cell_count *= max(size_t(1), reduce_size);
        }

        size_t num_cell = output_stride.size[0]*ts_output_size.size[0];
        
        int thread_per_cell = 1;
        while((thread_per_cell<<1) < reduce_cell_count){
            thread_per_cell=thread_per_cell<<1;
        }
        thread_per_cell = std::min(CUDA_NUM_THREADS, thread_per_cell);
        int cell_per_block = std::max(1, CUDA_NUM_THREADS/thread_per_cell);
        int share_size = CUDA_NUM_THREADS*sizeof(T);

        int num_block = ceilDiv<int>(num_cell, cell_per_block);
        num_block = min(num_block, kMaxGridNum);
        
        switch(pool_type){
            case PoolType::MAX:
            adaptive_pool_kernel<T, idleOp<T>, maxOp<T>, idleOp<T>><<< num_block, CUDA_NUM_THREADS, share_size, stream >>>(
                output, input,
                ts_input_size, input_stride,
                ts_output_size, output_stride,
                nb_dims, nb_reduce_dims, 
                cell_per_block, thread_per_cell,
                idleOp<T>(),
                maxOp<T>(),
                idleOp<T>(),
                num_cell);
            break;
            case PoolType::AVERAGE:
            adaptive_pool_kernel<T, idleOp<T>, sumOp<T>, divCountOp<T>><<< num_block, CUDA_NUM_THREADS, share_size, stream >>>(
                output, input,
                ts_input_size, input_stride,
                ts_output_size, output_stride,
                nb_dims, nb_reduce_dims, 
                cell_per_block, thread_per_cell,
                idleOp<T>(),
                sumOp<T>(),
                divCountOp<T>(),
                num_cell);
            break;
            default:
            break;
        }
                    
    }


    template void adaptive_pool<float>(float *output, const float* input, 
                                        int* input_dims, int* output_dims, int nb_dims,
                                        int nb_reduce_dims,
                                        PoolType pool_type,
                                        cudaStream_t stream);

}
}