#include <torch/extension.h>
#include <vector>

// cuda declarations
void basis_cuda_forward(int n, int m, int g, int k, int s, int p, const float* grid, const float* features, 
                        float* bases, const int* kernel_indices, const int* inp_idx, float* result);

/**
 * 
 * @param grid: (k, m, g) tensor of grid points
 * @param indice_pair_num: (k, 1)
 * @param indice_pairs: (2, k, max(indice_pair_num))
 * @param features: (p, m) p is number of points
 * @param g: number of grid parameters
 * @param k: number of kernel elements
 * @param s: spline order
 * 
 * 
*/
std::vector<torch::Tensor> basis_forward(torch::Tensor features, torch::Tensor grid, torch::Tensor indice_pairs, torch::Tensor indice_pair_num, int g, int k, int s) {

    grid = grid.contiguous();
    indice_pairs = indice_pairs.contiguous().permute({1, 2, 0}); // (k, max(indice_pair_num), 2)
    indice_pair_num = indice_pair_num.contiguous();

    int num_operations = indice_pair_num.sum().item<int>();
    int num_points = features.size(0);
    int num_dims = features.size(1);
    
    torch::Tensor result = torch::empty({num_operations, num_points, g}, torch::kFloat);
    torch::Tensor bases = torch::empty({num_operations, num_points, g}, torch::kFloat);
    torch::Tensor inp_idx = torch::empty({num_operations}, torch::kInt);
    torch::Tensor kernel_indices = torch::empty({num_operations}, torch::kInt);

    int current_index = 0;
    for (int i = 0; i < k; i++) {
        int num_ops = indice_pair_num[i].item<int>(); // how many valid operations for kernel
        if (num_ops > 0) {
          auto inp = indice_pairs.index({i, torch::indexing::Slice(torch::indexing::None, num_ops), 0});
          auto out = indice_pairs.index({i, torch::indexing::Slice(torch::indexing::None, num_ops), 1});

          inp_idx.index_put_({torch::indexing::Slice(current_index, current_index + num_ops)}, inp);
          kernel_indices.index_put_({torch::indexing::Slice(current_index, current_index + num_ops)}, i);
          current_index += num_ops;
        }
    }

    basis_cuda_forward(num_operations, num_dims, g, k, s, num_points, grid.data_ptr<float>(), features.data_ptr<float>(), 
                    bases.data_ptr<float>(), kernel_indices.data_ptr<int>(), inp_idx.data_ptr<int>(), result.data_ptr<float>());

    return {result};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &basis_forward, "Basis forward (CUDA)",
        py::arg("features"), py::arg("grid"), py::arg("indice_pairs"), 
        py::arg("indice_pair_num"), py::arg("g"), py::arg("k"), py::arg("s"));
}