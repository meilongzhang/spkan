from numba import cuda

@cuda.jit(device=True)
def base_matmul(bases, weights, out_features):
    for i in range(len(out_features)):
        sum = 0
        for j in range(len(weights[i])):
            sum += weights[i][j] * bases[j]
        
        cuda.atomic.add(out_features, i, sum)


@cuda.jit(device=True)
def grid_matmul(bases, weights, out_features):
    for i in range(len(out_features)):
        sum = 0
        for j in range(len(weights[i])):
            sum += weights[i][j] * bases[j // len(bases[0])][j % len(bases[0])]
        
        cuda.atomic.add(out_features, i, sum)
        
@cuda.jit(device=True)
def update_grid(x, grid, bases, temp, spline_order, spline_weights, out_channels, in_channels):
    kernel_idx = cuda.grid(1)
    b_splines(x, grid, bases, temp, spline_order)
    bases[:,:-4] # (in, coeff)
    spline_weights[kernel_idx] # (out, in x coeff)
    bases = bases.transpose(1, 0, 2)
    orig_coeff = spline_weights[kernel_idx].reshape(out_channels, in_channels, -1)

@cuda.jit(device=True)
def b_splines(x, grid, bases, temp, spline_order):
    kernel_idx = cuda.grid(1)
    for i in range(len(x)):
        for j in range(len(bases[i])):
            bases[i][j] = int((x[i] >= grid[kernel_idx][:,:-1][i][j]) & (x[i] < grid[kernel_idx][:,1:][i][j]))

    for k in range(1, spline_order + 1):
        #temp = np.zeros(len(x), 12 - k) # second value is uncertain

        for i in range(len(x)):
            for j in range(len(bases[i]) - 4 - k):
                temp[i][j] = (
                    (x[i] - grid[kernel_idx][:,:-(k+1)][i][j])
                    / (grid[kernel_idx][:,k:-1][i][j] - grid[kernel_idx][:,:-(k+1)][i][j])
                    * bases[i][:-1][j]
                ) + (
                    (grid[kernel_idx][:,k+1:][i][j] - x[i])
                    / (grid[kernel_idx][:,k+1:][i][j] - grid[kernel_idx][:,1:(-k)][i][j])
                    * bases[i][1:][j]
                )

        bases = temp

@cuda.jit
def test_conv(indice_pairs, indice_pair_num, features, activated, out_features, spline_weights, base_weights, grid, grid_size, spline_order, grid_eps, result, temp):
    kernel_idx = cuda.grid(1)
    if kernel_idx < 27:

      iopairs = indice_pairs[:,kernel_idx,:indice_pair_num[kernel_idx]]

      inp = iopairs[0, :]
      out = iopairs[1, :]
      for i in range(len(inp)):
          x = features[inp[i]]
          # update_grid(x, grid, result[kernel_idx], temp, spline_order, spline_weights, 5, 3)
          b_splines(x, grid, result[kernel_idx], temp, spline_order)

          grid_matmul(result[kernel_idx][:,:-4], spline_weights[kernel_idx], out_features[out[i]]) # (in, coeff)
          base_matmul(activated[inp[i]], base_weights[kernel_idx], out_features[out[i]])