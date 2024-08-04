#include <cuda_runtime.h>

/***
 * 
 * @param inp_idx: a 1d array of size n, where n is the number of operations. Each element is an input indice location for an operation
 * @param features: a 1d array of size p*m, where p is the number of points and m is the dimension of the input features
 * @param bases:  a 1d array of size nxmxg, where n is the number of operations, m is number of input feature dimensions, g is number of grid parameters
 * @param grid: a 1d array of size kxmxg, where k is the number of kernel elements, m is the number of input feature dimensions, g is the number of grid parameters
 * @param result: a 1d array of size nxmxg
 * @param g: number of parameters for each grid subsection
 * @param k: number of kernel elements
 * @param n: number of operations
 * @param m: dimension of input features
 * @param s: spline order
*/
__global__ void bsplineforwardKernel(int n, int m, int g, int k, int s, const float * __restrict__ grid, const float * __restrict__ features, float * bases, const int * __restrict__ kernel_indices, const int * __restrict__ inp_idx, float * result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n*m) {
        return;
    }

    int operation_idx = idx / m; // which operation
    int feature_idx = idx % m; // which feature dimension
    int point_idx = inp_idx[operation_idx] // which point
    int kernel_idx = kernel_indices[operation_idx] // which kernel
    float feature = features[point_idx * m + feature_idx] // which feature

    int gridstart = kernel_idx * m * g + feature_idx * g // the start of associated grid
    int gridend = gridstart + g // end of associated grid, not inclusive

    int basestart = idx * g // start of associated bases
    int baseend = basestart + g // end of associated bases, not inclusive

    /* Instantiate bases */
    for (int i = 0; i < g-1; i++) {
        bases[basestart + i] = (feature >= grid[gridstart + i]) && (feature < grid[gridstart + i + 1])
    }

    /* Loop for spline order */
    for (int c = 1; c < s + 1; c++) {     

        /* Loop for bases elements */ 
        for (int i = 0; i < g - (c + 1); i++) {
            result[basestart + i] = ((feature - grid[gridstart + i]) / (grid[gridstart + c + i] - grid[gridstart + i]) * bases[basestart + i])
                + ((grid[gridstart + c + i + 1] - feature) / (grid[gridstart + c + i + 1] - grid[gridstart + 1]) * bases[basestart + i + 1])
        }

        if (c == s) {
            break;
        }
        
        for (i = 0; i < g - (c + 1); i++) {
            bases[basestart + i] = result[basestart + i]
        }
    }
}

int main(int argc, char **argv) {
    return 0;
}