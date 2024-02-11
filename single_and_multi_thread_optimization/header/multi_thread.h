#include <pthread.h>
#include <immintrin.h>
#include "cstring"
#include <thread>
#include <algorithm>

#define COLUMN_PADDING 8
#define SINGLE_THREAS 4U
#define MAX_THREADS 64U

#define FAST

#pragma GCC diagnostic ignored "-Wregister"

// CODE for dynamically selecting threads
// Select NO_OF_HYPER_THREAD - 1
// if unable to fetch threads then set NUM_OF_THREADS to 1
// and MAX THREADS is 128
const int NUM_OF_THREADS = std::min(std::max(std::thread::hardware_concurrency()-1, SINGLE_THREAS), MAX_THREADS);

typedef struct ThreadData 
{
    int input_row;
    int input_col;
    int *input=nullptr; 
    int kernel_row; 
    int kernel_col; 
    int *kernel=nullptr;
    int output_row; 
    int output_col; 
    long long unsigned int *output = nullptr; 
    int index;
} ThreadData;

void* threadRowsCalculate(void* data)
{
    ThreadData *threadData = (ThreadData *)data;
    register int input_row = threadData->input_row;
    register int input_col = threadData->input_col;
    register int *input = threadData->input;
    register int kernel_row = threadData->kernel_row;
    register int kernel_col = threadData->kernel_col;
    register int *kernel = threadData->kernel;
    register int output_row = threadData->output_row;
    register int output_col = threadData->output_col;
    register long long unsigned int *output = threadData->output;
    register int index = threadData->index;

    register int output_index, kernel_index, input_i, input_base, input_j;
    register int kernel_row_ = kernel_row << 1;
    register int kernel_col_ = kernel_col << 1;

    int output_col_ = ((output_col >> 3) << 3) + 8;
    long long unsigned int* output_ = (long long unsigned int*)malloc(output_col_ * sizeof(long long unsigned int));

    __m256i offset = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    
    for(int i = index; i< output_row; i+=NUM_OF_THREADS){
        // output_index = i * output_col;
        output_index = 0;
#ifdef FAST
    #pragma unroll
#endif
        for (int j = 0; j < output_col; j += 8)
        {
            __m256i start_four_result = _mm256_setzero_si256();
            __m256i last_four_result = _mm256_setzero_si256();
            kernel_index = 0;
#ifdef FAST
    #pragma unroll
#endif
            for (int kernel_i = 0; kernel_i < kernel_row_; kernel_i += 2)
            {
                input_i = i + kernel_i;
                input_i = input_i >= input_row ? (input_i - input_row) : input_i;
                input_base = input_i * input_col;
#ifdef FAST
    #pragma unroll
#endif
                for (int kernel_j = 0; kernel_j < kernel_col_; kernel_j += 2)
                {
                    input_j = j + kernel_j;
                    input_j = input_j >= input_col ? (input_j - input_col) : input_j;
                    __m256i total;
                    if (input_j + 7 < input_col)
                    {
                        // __builtin_prefetch(&input[input_base +  input_j + 128]);
                        total = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i *)&input[input_base + input_j]), _mm256_set1_epi32(kernel[kernel_index++]));
                    }
                    else
                    {
                        __m256i mod_index = _mm256_add_epi32(_mm256_set1_epi32(input_base),_mm256_add_epi32(
                            _mm256_mullo_epi32(
                                _mm256_cmpgt_epi32(_mm256_add_epi32(_mm256_set1_epi32(input_j), offset), _mm256_set1_epi32(input_col-1)),
                                _mm256_set1_epi32(input_col)
                            ),
                            _mm256_add_epi32(_mm256_set1_epi32(input_j), offset)
                        ));
                        total = _mm256_mullo_epi32(_mm256_set_epi32(
                            input[((int*)&mod_index)[0]],
                            input[((int*)&mod_index)[1]],
                            input[((int*)&mod_index)[2]],
                            input[((int*)&mod_index)[3]],
                            input[((int*)&mod_index)[4]],
                            input[((int*)&mod_index)[5]],
                            input[((int*)&mod_index)[6]],
                            input[((int*)&mod_index)[7]]),
                            _mm256_set1_epi32(kernel[kernel_index++])
                        );
                    }
                    start_four_result = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extractf128_si256(total, 1)), start_four_result);
                    last_four_result = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(total)), last_four_result);
                }
            }
            _mm256_storeu_si256((__m256i *)&output_[output_index], last_four_result);
            _mm256_storeu_si256((__m256i *)&output_[output_index + 4], start_four_result);
            output_index += 8;
        }
        std::memcpy(&output[i*output_col], output_, sizeof(long long unsigned int)*output_col);  
    }
    return nullptr;
}


void multiThread( 
    int input_row, 
    int input_col,
    int *input, 
    int kernel_row, 
    int kernel_col, 
    int *kernel,
    int output_row, 
    int output_col, 
    long long unsigned int *output 
) 
{
    ThreadData *threadingData = (ThreadData *)malloc(sizeof(ThreadData) * NUM_OF_THREADS);
    pthread_t threadPool[NUM_OF_THREADS];
    
    for(int index = 0; index < NUM_OF_THREADS; index++){
            threadingData[index].input_row = input_row;
            threadingData[index].input_col = input_col;
            threadingData[index].input = input;
            threadingData[index].kernel_row = kernel_row;
            threadingData[index].kernel_col = kernel_col;
            threadingData[index].kernel = kernel;
            threadingData[index].output_row = output_row;
            threadingData[index].output_col = output_col;
            threadingData[index].output = output;
            threadingData[index].index = index;
    }


    for(int i = 0; i < NUM_OF_THREADS; i++){
        int pid = pthread_create(&threadPool[i], NULL, &threadRowsCalculate, (void *)&threadingData[i]);
        if (pid != 0){
            std::cout << "Unable to create Threads" << endl;
            exit(1);
        }
    }

    for(int i = 0; i < NUM_OF_THREADS; i++)
        pthread_join(threadPool[i], nullptr);
}
