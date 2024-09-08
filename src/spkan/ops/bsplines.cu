#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/extension.h>
#include <vector>

/***
 * 
 * @param inp_idx: a 1d array of size n, where n is the number of operations. Each element is an input indice location for an operation
 * @param features: a 1d array of size p*m, where p is the number of points and m is the dimension of the input features
 * @param bases:  a 1d array of size nxmxg, where n is the number of operations, m is number of input feature dimensions, g is number of grid parameters
 * @param grid: a 1d array of size kxmxg, where k is the number of kernel elements, m is the number of input feature dimensions, g is the number of grid parameters
 * @param result: a 1d array of size nxmxg
 * @param kernel_indices: a 1d array of size n, where n is the number of operations. Each element is a kernel indice location for an operation
 * @param g: number of parameters for each grid subsection
 * @param k: number of kernel elements
 * @param n: number of operations
 * @param m: dimension of input features
 * @param s: spline order
 * 
 * @param o: output feature dimension
 * @param b: dimension of bases for use with spline weights
 * @param spline_weights: 1d array of size k*o*m*b
 * @param base_weights: 1d array of shape k*o*m
 * @param spline_outputs: num_outxo
 * @param out_idx: 1d array of size n. each element is an output indice location for an operation
 * @param activated_features: 1d array of size p*m, features passed through specified activation function
 * 
 * 
*/

//maybe can add more __restrict__ here
__global__ void basis_cuda_forward_kernel(int n, int m, int g, int k, int s, 
                                    const float * __restrict__ grid, const float * __restrict__ features, 
                                    float * __restrict__ bases, const int * __restrict__ kernel_indices, 
                                    const int * __restrict__ inp_idx, float * __restrict__ result, const int * __restrict__ out_idx,
                                    const float * __restrict__ activated_features, const float * __restrict__ spline_weights,
                                    const float * __restrict__ base_weights, float * __restrict__ spline_output, int o, int b, int num_out) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx >= n*m) {
        return;
    }

    int operation_idx = idx / m; // which operation
    int feature_idx = idx % m; // which input feature dimension

    int point_idx = inp_idx[operation_idx]; // which point
    int out_point_idx = out_idx[operation_idx]; // which output point
    int kernel_idx = kernel_indices[operation_idx]; // which kernel
    float feature = features[point_idx * m + feature_idx]; // which feature
    float activated_feature = activated_features[point_idx * m + feature_idx]; // which activated feature
    
    int gridstart = kernel_idx * m * g + feature_idx * g; // the start of associated grid

    int basestart = operation_idx * m * g + feature_idx * g; // start of associated bases

    /* Instantiate bases */
    //#pragma unroll
    for (int i = 0; i < g-1; i++) {
        bases[basestart + i] = (feature >= grid[gridstart + i]) && (feature < grid[gridstart + i + 1]);
    }

    /* Loop for spline order */
    for (int c = 1; c < s + 1; c++) {     

        /* Loop for bases elements */ 
        //#pragma unroll 2
        for (int i = 0; i < g - (c + 1); i++) {
            result[basestart + i] = ((feature - grid[gridstart + i]) / 
                (grid[gridstart + c + i] - grid[gridstart + i]) * bases[basestart + i])
                + ((grid[gridstart + c + i + 1] - feature) / 
                (grid[gridstart + c + i + 1] - grid[gridstart + 1 + i]) * bases[basestart + i + 1]);
        }
        
        //#pragma unroll
        for (int i = 0; i < g - (c + 1); i++) {
            bases[basestart + i] = result[basestart + i];
        }
    }

    /* Loop for spline output */
    for (int i = 0; i < o; i++) { // increment output dimension
        int sw_start = kernel_idx * o * m + i * m + feature_idx;
        float temp = activated_feature * base_weights[sw_start];
        //atomicAdd(&spline_output[out_point_idx * o + i], activated_feature * base_weights[sw_start]);
        for (int j = 0; j < b; j++) {
            // if (out_point_idx >= num_out) {
            //     *status = out_point_idx;
            //     return;
            // }
            //atomicExch(last_base, basestart + j);
            temp = temp + result[basestart + j] * spline_weights[sw_start * b + j];

            //atomicAdd(&spline_output[out_point_idx * o + i], result[basestart + j] * spline_weights[sw_start * b + j]);
        }
        atomicAdd(&spline_output[out_point_idx * o + i], temp);
    }
}

/**
 * 
 * @param n: number of operations
 * @param m: dimension of input features
 * @param g: number of parameters for each grid subsection
 * @param k: number of kernel elements
 * @param s: spline order
 * @param o: output feature dimension
 * @param b: dimension of bases for use with spline weights
 * @param num_out: number of output points
 * @param grad_output: 1d array of size num_outxo
 * @param activated: 1d array of size p*m, features passed through specified activation function
 * @param spline_weights: 1d array of size k*o*m*b
 * @param base_weights: 1d array of shape k*o*m
 * @param bases: 1d array of size nxmxg
 * @param inp_idx: 1d array of size n, each element is an input indice location for an operation
 * @param out_idx: 1d array of size n, each element is an output indice location for an operation
 * @param kernel_indices: 1d array of size n, each element is the specified kernel element for an operation
 * @param grad_input_feature: 1d array of size p*m
 * @param grad_activated: 1d array of size p*m
 * @param grad_spline_weights: 1d array of size k*o*m*b
 * @param grad_base_weights: 1d array of size k*o*m
 * 
 * 
 */
__global__ void basis_cuda_backward_kernel(int n, int m, int g, int k, int s, int o, int b, int num_out, 
                                    const float * __restrict__ grad_output, const float * __restrict__ activated, 
                                    const float * __restrict__ spline_weights, const float * __restrict__ base_weights, 
                                    const float * __restrict__ bases, const int * __restrict__ inp_idx, 
                                    const int * __restrict__ out_idx, const int * __restrict__ kernel_indices, 
                                    float * __restrict__ grad_input_feature, float * __restrict__ grad_activated, 
                                    float * __restrict__ grad_spline_weights, float * __restrict__ grad_base_weights) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx >= n*m) {
        return;
    }

    int operation_idx = idx / m; // which operation
    int feature_idx = idx % m; // which input feature dimension
    int point_idx = inp_idx[operation_idx]; // which point
    int out_point_idx = out_idx[operation_idx]; // which output point
    int kernel_idx = kernel_indices[operation_idx]; // which kernel

    float activated_feature = activated[point_idx * m + feature_idx]; // which activated feature

    int grad_output_start = out_point_idx * o; // o output dimensions
    //int grad_input_feature_start = point_idx * m; // m input dimensions

    for (int i = 0; i < o; i++) { // iterate for output dimension
        int sw_start = kernel_idx * o * m + i * m + feature_idx;
        //float grad_activated_temp = grad_output[grad_output_start + i] * base_weights[sw_start];
        //float grad_base_weights_temp = grad_output[grad_output_start + i] * activated_feature;
        
        //atomicAdd(&grad_activated[point_idx * m + feature_idx], grad_output[grad_output_start + i] * base_weights[sw_start]);
        
        atomicAdd(&grad_base_weights[sw_start], grad_output[grad_output_start + i] * activated_feature);

        //float grad_spline_weights_temp = 0;

        for (int j = 0; j < b; j++) {
            //grad_input_feature_temp = grad_input_feature_temp + grad_output[grad_output_start + i] * spline_weights[sw_start * b + j];
            atomicAdd(&grad_spline_weights[sw_start * b + j], grad_output[grad_output_start + i] * bases[operation_idx * m * g + feature_idx * g + j]);
        }

        // atomicAdd(&grad_input_feature[grad_input_feature_start + feature_idx], grad_input_feature_temp);
        
    }
}


void basis_cuda_forward(int n, int m, int g, int k, int s, int p, const float * __restrict__ grid, 
                        const float * __restrict__ features, float * __restrict__ bases, 
                        const int * __restrict__ kernel_indices, const int * __restrict__ inp_idx, 
                        float * __restrict__ result, const int * __restrict__ out_idx, const float * __restrict__ activated_features,
                        const float * __restrict__ spline_weights, const float * __restrict__ base_weights,
                        float * __restrict__ spline_output, int o, int b, int num_out) {

    // float * d_bases;
    // int * __restrict__ d_kernel_indices;
    // int * __restrict__ d_inp_idx;
    // float * d_result;
    // int * __restrict__ d_out_idx;
    // float * d_spline_output;

    cudaError_t err;

    // err=cudaMalloc((void**)&d_bases, n * m * g * sizeof(float));
    //     if (err != cudaSuccess) {std::cout << "cudaMalloc d_bases failed: " << cudaGetErrorString(err) << std::endl; return;}
    // err=cudaMalloc((void**)&d_kernel_indices, n * sizeof(int));
    //     if (err != cudaSuccess) {std::cout << "cudaMalloc d_kernel_indices failed: " << cudaGetErrorString(err) << std::endl; return;}
    // err=cudaMalloc((void**)&d_inp_idx, n * sizeof(int));
    //     if (err != cudaSuccess) {std::cout << "cudaMalloc d_inp_idx failed: " << cudaGetErrorString(err) << std::endl; return;}
    // err=cudaMalloc((void**)&d_result, n * m * g * sizeof(float));
    //     if (err != cudaSuccess) {std::cout << "cudaMalloc d_result failed: " << cudaGetErrorString(err) << std::endl; return;}
    // err=cudaMalloc((void**)&d_out_idx, n * sizeof(int));
    //     if (err != cudaSuccess) {std::cout << "cudaMalloc d_out_idx failed: " << cudaGetErrorString(err) << std::endl; return;}
    // err=cudaMalloc((void**)&d_spline_output, num_out * o * sizeof(float));
    //     if (err != cudaSuccess) {std::cout << "cudaMalloc d_spline_output failed: " << cudaGetErrorString(err) << std::endl; return;}
    
    // cudaMemcpy(d_bases, bases, n * m * g * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_kernel_indices, kernel_indices, n * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_inp_idx, inp_idx, n * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_result, result, n * m * g * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_out_idx, out_idx, n * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_spline_output, spline_output, num_out * o * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 1024; // how many threads per block
    int gridSize = (n * m + blockSize - 1) / blockSize; // how many blocks
    // basis_cuda_forward_kernel<<<gridSize, blockSize>>>(n, m, g, k, s, grid, features, d_bases, 
    //                                                     d_kernel_indices, d_inp_idx, d_result, d_out_idx,
    //                                                     activated_features, spline_weights, base_weights, 
    //                                                     d_spline_output, o, b, num_out);

    basis_cuda_forward_kernel<<<gridSize, blockSize>>>(n, m, g, k, s, grid, features, bases, 
                                                        kernel_indices, inp_idx, result, out_idx,
                                                        activated_features, spline_weights, base_weights, 
                                                        spline_output, o, b, num_out);

    err = cudaGetLastError();
    if (err != cudaSuccess) {std::cout << "basis_cuda_forward_kernel launch failed: " << cudaGetErrorString(err) << std::endl; return;}

    // cudaMemcpy(result, d_result, n * m * g * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(result, d_result, n * m * g * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(spline_output, d_spline_output, num_out * o * sizeof(float), cudaMemcpyDeviceToHost);

    // cudaFree(d_bases);
    // cudaFree(d_kernel_indices);
    // cudaFree(d_inp_idx);
    // cudaFree(d_result);
    // cudaFree(d_out_idx);
    // cudaFree(d_spline_output);

    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {std::cout << "basis_cuda_forward failed: " << cudaGetErrorString(err) << std::endl; return;}
}

void basis_cuda_backward(int n, int m, int g, int k, int s, int p, int o, int b, int num_out,
                         const float * __restrict__ grad_output, 
                         const float * __restrict__ activated, const float * __restrict__ spline_weights, 
                         const float * __restrict__ base_weights, const float * __restrict__ bases, 
                         const int * __restrict__ inp_idx, const int * __restrict__ out_idx, 
                         const int * __restrict__ kernel_indices, float * __restrict__ grad_input_feature, 
                         float * __restrict__ grad_activated, float * __restrict__ grad_spline_weights, 
                         float * __restrict__ grad_base_weights) {
    

    cudaError_t err;

    int blockSize = 1024; // how many threads per block
    int gridSize = (n * m + blockSize - 1) / blockSize; // how many blocks
    basis_cuda_backward_kernel<<<gridSize, blockSize>>>(n, m, g, k, s, o, b, num_out, grad_output, activated, spline_weights, base_weights, 
                                                        bases, inp_idx, out_idx, kernel_indices, grad_input_feature, 
                                                        grad_activated, grad_spline_weights, grad_base_weights);

    err = cudaGetLastError();
    if (err != cudaSuccess) {std::cout << "basis_cuda_backward_kernel launch failed: " << cudaGetErrorString(err) << std::endl; return;}

    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {std::cout << "basis_cuda_backward failed: " << cudaGetErrorString(err) << std::endl; return;}
}

int main(int argc, char **argv) {
    return 0;
}