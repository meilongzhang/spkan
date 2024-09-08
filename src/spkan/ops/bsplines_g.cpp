#include <torch/extension.h>
#include <vector>

// cuda declarations
void basis_cuda_forward(int n, int m, int g, int k, int s, int p, const float* grid, const float* features, 
                        float* bases, const int* kernel_indices, const int* inp_idx, float* result,
                        const int* out_idx, const float * activated_features, const float * spline_weights, 
                        const float * base_weights, float * spline_output, int o, int b, int num_out);


void basis_cuda_backward(int n, int m, int g, int k, int s, int p, int o, int b, int num_out,
                         const float* grad_output, const float* activated, const float* spline_weights, 
                         const float* base_weights, const float* bases, const int* inp_idx, 
                         const int* out_idx, const int* kernel_indices, 
                         float* grad_input_feature, float* grad_activated, 
                         float* grad_spline_weights, float* grad_base_weights);
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
 * @param activated_features: (p, m)
 * @param spline_weights: (k, o, m * b)
 * @param base_weights: (k, o, m)
 * 
 * What do I want:
 * spline_output: (n, o)
 * 
 * 
*/
std::vector<torch::Tensor> basis_forward(torch::Tensor features, torch::Tensor grid, torch::Tensor indice_pairs, 
                                      torch::Tensor indice_pair_num, int g, int k, int s,
                                      torch::Tensor activated_features, torch::Tensor spline_weights, 
                                      torch::Tensor base_weights, int num_out) {

    features = features.contiguous();
    grid = grid.contiguous();
    indice_pairs = indice_pairs.contiguous().permute({1, 2, 0}); // (k, max(indice_pair_num), 2)
    indice_pair_num = indice_pair_num.contiguous();
    spline_weights = spline_weights.contiguous();
    base_weights = base_weights.contiguous();
    activated_features = activated_features.contiguous();

    int num_operations = indice_pair_num.sum().item<int>();
    int num_points = features.size(0);
    int num_dims = features.size(1);
    int out_dim = spline_weights.size(1); // output feature dimension, o
    int basis_dim = g - s - 1; // dimension of final result, b
    
    torch::Tensor result = torch::empty({num_operations, num_dims, g}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor bases = torch::empty({num_operations, num_dims, g}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor inp_idx = torch::empty({num_operations}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    torch::Tensor out_idx = torch::empty({num_operations}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    torch::Tensor kernel_indices = torch::empty({num_operations}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    torch::Tensor spline_output = torch::zeros({num_out, out_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // torch::Tensor result = torch::empty({num_operations, num_dims, g}, torch::kFloat32);
    // torch::Tensor bases = torch::empty({num_operations, num_dims, g}, torch::kFloat32);
    // torch::Tensor inp_idx = torch::empty({num_operations}, torch::kInt);
    // torch::Tensor out_idx = torch::empty({num_operations}, torch::kInt);
    // torch::Tensor kernel_indices = torch::empty({num_operations}, torch::kInt);
    // torch::Tensor spline_output = torch::zeros({num_out, out_dim}, torch::kFloat32);


    int current_index = 0;
    for (int i = 0; i < k; i++) {
        int num_ops = indice_pair_num[i].item<int>(); // how many valid operations for kernel
        if (num_ops > 0) {
          auto inp = indice_pairs.index({i, torch::indexing::Slice(torch::indexing::None, num_ops), 0});
          auto out = indice_pairs.index({i, torch::indexing::Slice(torch::indexing::None, num_ops), 1});

          inp_idx.index_put_({torch::indexing::Slice(current_index, current_index + num_ops)}, inp);
          out_idx.index_put_({torch::indexing::Slice(current_index, current_index + num_ops)}, out);
          kernel_indices.index_put_({torch::indexing::Slice(current_index, current_index + num_ops)}, i);
          current_index += num_ops;
        }
    }

    if (num_operations * num_dims > 0) {
        basis_cuda_forward(num_operations, num_dims, g, k, s, num_points, grid.data_ptr<float>(), features.data_ptr<float>(), 
                      bases.data_ptr<float>(), kernel_indices.data_ptr<int>(), inp_idx.data_ptr<int>(), result.data_ptr<float>(),
                      out_idx.data_ptr<int>(), activated_features.data_ptr<float>(), spline_weights.data_ptr<float>(), base_weights.data_ptr<float>(),
                      spline_output.data_ptr<float>(), out_dim, basis_dim, num_out);
    }
    
    return {spline_output, result, inp_idx, out_idx, kernel_indices};
}

std::vector<torch::Tensor> basis_backward(int g, int k, int s, int num_out,
                                       torch::Tensor grad_output, torch::Tensor activated, 
                                       torch::Tensor spline_weights, torch::Tensor base_weights, 
                                       torch::Tensor bases, torch::Tensor inp_idx, 
                                       torch::Tensor out_idx, torch::Tensor kernel_indices) {
    grad_output = grad_output.contiguous();
    activated = activated.contiguous();
    spline_weights = spline_weights.contiguous();
    base_weights = base_weights.contiguous();
    bases = bases.contiguous();
    inp_idx = inp_idx.contiguous();
    out_idx = out_idx.contiguous();
    kernel_indices = kernel_indices.contiguous();

    int num_operations = inp_idx.size(0); // n
    int num_dims = activated.size(1); // m
    int num_points = activated.size(0); // p
    int out_dim = spline_weights.size(1); // o
    int basis_dim = g - s - 1; // b

    torch::Tensor grad_input_feature = torch::zeros_like(activated);
    torch::Tensor grad_activated = torch::zeros_like(activated);
    torch::Tensor grad_spline_weights = torch::zeros_like(spline_weights);
    torch::Tensor grad_base_weights = torch::zeros_like(base_weights);

    basis_cuda_backward(num_operations, num_dims, g, k, s, num_points, out_dim, basis_dim, num_out,
                        grad_output.data_ptr<float>(), activated.data_ptr<float>(), 
                        spline_weights.data_ptr<float>(), base_weights.data_ptr<float>(), 
                        bases.data_ptr<float>(), inp_idx.data_ptr<int>(), 
                        out_idx.data_ptr<int>(), kernel_indices.data_ptr<int>(), 
                        grad_input_feature.data_ptr<float>(), grad_activated.data_ptr<float>(), 
                        grad_spline_weights.data_ptr<float>(), grad_base_weights.data_ptr<float>());

    return {grad_input_feature, grad_activated, grad_spline_weights, grad_base_weights};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &basis_forward, "Basis forward (CUDA)",
        py::arg("features"), py::arg("grid"), py::arg("indice_pairs"), 
        py::arg("indice_pair_num"), py::arg("g"), py::arg("k"), py::arg("s"),
        py::arg("activated_features"), py::arg("spline_weights"), py::arg("base_weights"),
        py::arg("num_out"));

  m.def("backward", &basis_backward, "Basis backward (CUDA)",
        py::arg("g"), py::arg("k"), py::arg("s"), py::arg("num_out"),
        py::arg("grad_output"), py::arg("activated"), py::arg("spline_weights"),
        py::arg("base_weights"), py::arg("bases"), py::arg("inp_idx"),
        py::arg("out_idx"), py::arg("kernel_indices"));
}