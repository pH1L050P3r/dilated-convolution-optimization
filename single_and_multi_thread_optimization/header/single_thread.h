#include <immintrin.h>
#include <algorithm>
#include "cstring"
#include "cstdlib"

#define FAST

void singleThread(
    int input_row,
    int input_col,
    int *input,
    int kernel_row,
    int kernel_col,
    int *kernel,
    int output_row,
    int output_col,
    long long unsigned int *output)
{

#ifdef FAST
    // variable  for reducing operation
    #pragma GCC diagnostic ignored "-Wregister"
    int output_index;
    register int kernel_index;
    register int input_i, input_j, input_base;
    register int kernel_row_ = kernel_row << 1;
    register int kernel_col_ = kernel_col << 1;
    register int output_col_ = ((output_col >> 3) << 3);
    // end
#else
    int output_index;
    int kernel_index;
    int input_i, input_j, input_base;
    int kernel_row_ = kernel_row << 1;
    int kernel_col_ = kernel_col << 1;
    int output_col_ = ((output_col >> 3) << 3);
#endif
    __m256i offset = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i v_input_col = _mm256_set1_epi32(input_col);
    __m256i v_input_col_cmp = _mm256_set1_epi32(input_col-1);
    __m256i calculate_offset;
    __m256i mod_index;

#ifdef FAST
    #pragma unroll
#endif
    for (int i = 0; i < output_row; i++)
    {
        output_index = i * output_col;
#ifdef FAST
    #pragma unroll
#endif
        for (int j = 0; j < output_col_; j += 8)
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
                __m256i v_input_base = _mm256_set1_epi32(input_base);
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
                        total = _mm256_mullo_epi32(
                            _mm256_loadu_si256((__m256i *)&input[input_base + input_j]), 
                            _mm256_set1_epi32(kernel[kernel_index++])
                        );
                    }
                    else
                    {
                        calculate_offset = _mm256_add_epi32(_mm256_set1_epi32(input_j), offset);
                        mod_index = _mm256_add_epi32(v_input_base,_mm256_add_epi32(
                            _mm256_mullo_epi32(_mm256_cmpgt_epi32(calculate_offset, v_input_col_cmp),v_input_col),
                            calculate_offset
                        ));

                        int* v_index_add = (int *)&mod_index;
                        total = _mm256_mullo_epi32(_mm256_set_epi32(
                            input[v_index_add[0]],
                            input[v_index_add[1]],
                            input[v_index_add[2]],
                            input[v_index_add[3]],
                            input[v_index_add[4]],
                            input[v_index_add[5]],
                            input[v_index_add[6]],
                            input[v_index_add[7]]),
                            _mm256_set1_epi32(kernel[kernel_index++])
                        );
                    }
                    start_four_result = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extractf128_si256(total, 1)), start_four_result);
                    last_four_result = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(total)), last_four_result);
                }
            }
            _mm256_storeu_si256((__m256i*)&output[output_index], last_four_result);
            _mm256_storeu_si256((__m256i*)&output[output_index+4], start_four_result);
            output_index += 8;
        }
        if(output_col_ < output_col)
        {
            __m256i start_four_result = _mm256_setzero_si256();
            __m256i last_four_result = _mm256_setzero_si256();
            kernel_index = 0;
            for (int kernel_i = 0; kernel_i < kernel_row_; kernel_i += 2)
            {
                input_i = i + kernel_i;
                input_i = input_i >= input_row ? (input_i - input_row) : input_i;
                input_base = input_i * input_col;
                __m256i v_input_base = _mm256_set1_epi32(input_base);
                for (int kernel_j = 0; kernel_j < kernel_col_; kernel_j += 2)
                {
                    input_j = output_col_ + kernel_j;
                    input_j = input_j >= input_col ? (input_j - input_col) : input_j;
                    __m256i total;
                    if (input_j + 7 < input_col)
                    {
                        total = _mm256_mullo_epi32(
                            _mm256_loadu_si256((__m256i *)&input[input_base + input_j]), 
                            _mm256_set1_epi32(kernel[kernel_index++])
                        );
                    }
                    else
                    {
                        calculate_offset = _mm256_add_epi32(_mm256_set1_epi32(input_j), offset);
                        mod_index = _mm256_add_epi32(v_input_base,_mm256_add_epi32(
                            _mm256_mullo_epi32(_mm256_cmpgt_epi32(calculate_offset, v_input_col_cmp),v_input_col),
                            calculate_offset
                        ));

                        int* v_index_add = (int *)&mod_index;
                        total = _mm256_mullo_epi32(_mm256_set_epi32(
                            input[v_index_add[0]],
                            input[v_index_add[1]],
                            input[v_index_add[2]],
                            input[v_index_add[3]],
                            input[v_index_add[4]],
                            input[v_index_add[5]],
                            input[v_index_add[6]],
                            input[v_index_add[7]]),
                            _mm256_set1_epi32(kernel[kernel_index++])
                        );
                    }
                    start_four_result = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extractf128_si256(total, 1)), start_four_result);
                    last_four_result = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(total)), last_four_result);
                }
            }
            if (output_col_ + 3 < output_col) {
                _mm256_storeu_si256((__m256i*)&output[output_index], last_four_result);
                if(output_col_+4 < output_col) output[output_index+4] = start_four_result[0];
                if(output_col_+5 < output_col) output[output_index+5] = start_four_result[1];
                if(output_col_+6 < output_col) output[output_index+6] = start_four_result[2];
            } else {
                output[output_index] = last_four_result[0];
                if(output_col_+1 < output_col) output[output_index+1] = last_four_result[1];
                if(output_col_+2 < output_col) output[output_index+2] = last_four_result[2];
            }
        } 
    }
}
