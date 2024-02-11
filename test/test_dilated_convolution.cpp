#include <iostream>
#include <string>

// Perform dilated convolution on a 2D matrix
void dilated_convolution(   int input_row, 
                            int input_col,
                            int *input, 
                            int kernel_row, 
                            int kernel_col, 
                            int *kernel, 
                            int *output ) {

    int output_row = input_row - kernel_row + 1;
    int output_col = input_col - kernel_col + 1;
    
    for(int output_i = 0; output_i< output_row; output_i++)
    {
        for(int output_j = 0; output_j< output_col; output_j++)
        {
            std::string s = "";
            for(int kernel_i = 0; kernel_i< kernel_row; kernel_i++)
            {
                for(int kernel_j = 0; kernel_j< kernel_col; kernel_j++)
                {
                    int input_i = (output_i + 2*kernel_i) % input_row;
                    int input_j = (output_j + 2*kernel_j) % input_col;
                    output[output_i * output_col + output_j] += 
                                    input[input_i*input_col +input_j] * kernel[kernel_i*kernel_col +kernel_j];
                    // store the positions of the input matrix and kernel that are used to calculate the output matrix
                    // s += input[input_i][input_j] * kernel[kernel_i][kernel_j]
                    s += "input[" + std::to_string(input_i) + "][" + std::to_string(input_j) + "] * kernel[" + std::to_string(kernel_i) + "][" + std::to_string(kernel_j) + "] + ";
                }
            }
            std::cout << "output[" << output_i << "][" << output_j << "] = " << s.substr(0, s.length()-3) << std::endl;
        }
    }

}

int main() {
    // Define input matrix
    int input[5*5] = {  1, 2, 3, 4, 5, 
                        6, 7, 8, 9, 10, 
                        11, 12, 13, 14, 15, 
                        16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25};

    // Define kernel matrix
    int kernel[3*3] = {   1, 2, 1,
                          2, 4, 2,
                          1, 2, 1};

    // Define output matrix
    int output[3*3] = {0}; 
    // Perform dilated convolution 
    dilated_convolution(5, 5, input, 3, 3, kernel, output);

    // Print output matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << output[i*3+j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}