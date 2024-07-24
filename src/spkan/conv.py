from spconv.pytorch import ops
from spconv.pytorch.core import expand_nd
from spconv.core import ConvAlgo
import math
import spconv.pytorch as spconv
from spconv.pytorch.utils import PointToVoxel
import numpy as np
import torch.nn.functional as F
import torch
from numba import cuda
from spconv.pytorch.modules import SparseModule
from typing import List, Optional, Tuple, Union
from functools import reduce
import operator
import sys

if __name__ == "__main__" and __package__ is None:
    import os
    import sys
    currentdir = os.path.dirname(os.path.abspath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    import cuda_tools
else:
    try:
        from spkan import cuda_tools
    except:
         import cuda_tools

array = np.array
float32 = np.float32

if sys.version_info[1] < 8:
    __PYTHON38__ = False
else:
    __PYTHON38__ = True

class SparseKANConv3d(SparseModule):
      """
      A pure Pytorch version of SparseKANConv3D. Offers Sparse 3D Convolution with Kolmogorov-Arnold Networks
      """

      def __init__(self,
                   ndim: int,
                   in_channels: int,
                   out_channels: int,
                   kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
                   stride: Union[int, List[int], Tuple[int, ...]] = 1,
                   padding: Union[int, List[int], Tuple[int, ...]] = 0,
                   dilation: Union[int, List[int], Tuple[int, ...]] = 1,
                   groups=1,
                   bias: bool = False,
                   subm: bool = False,
                   output_padding: Union[int, List[int], Tuple[int, ...]] = 0,
                   transposed: bool = False,
                   grid_size=3,
                   spline_order=3,
                   grid_range=[-1, 1],
                   grid_eps=0.02,
                   base_activation = torch.nn.SiLU,
                   device='cpu',
                   use_numba = False,
                   adaptive = False):
            super(SparseKANConv3d, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels

            self.kernel_size = expand_nd(ndim, kernel_size) if type(kernel_size) is int else kernel_size
            self.stride = expand_nd(ndim, stride) if type(stride) is int else stride
            self.padding = expand_nd(ndim, padding) if type(padding) is int else padding
            self.dilation = expand_nd(ndim, dilation) if type(dilation) is int else dilation
            self.output_padding = expand_nd(ndim, output_padding) if type(output_padding) is int else output_padding

            if __PYTHON38__:
                self.num_kernel_elems = math.prod(self.kernel_size)
            else:
                self.num_kernel_elems = reduce(operator.mul, self.kernel_size)

            self.subm = subm
            self.transposed = transposed
            self.device = device

            self.grid_size = grid_size
            self.spline_order = spline_order
            self.base_activation = base_activation()
            self.grid_eps = grid_eps
            self.cud = use_numba
            self.bias = bias
            self.lambda_value = 1e-5
            self.adaptive = adaptive

            num_elements = self.num_kernel_elems * in_channels
            h = (grid_range[1] - grid_range[0]) / grid_size

            # this grid is shared resource for all kernel elements
            self.grid = (
                (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
                )
                .expand(num_elements, -1)
                .contiguous()
            ).reshape((self.num_kernel_elems,in_channels,-1)).to(device) # 27 kernel locations, 3 input channels, 5 kernels (output channels), 12 bspline parameters
            #self.register_buffer("grid", self.grid)
            self.base_weights = torch.nn.Parameter(torch.Tensor(self.num_kernel_elems, out_channels, in_channels)).to(device)
            self.spline_weights = torch.nn.Parameter(torch.Tensor(self.num_kernel_elems, out_channels, in_channels * (grid_size + spline_order))).to(device)

            self.scale_base = 1.0
            self.scale_noise = 0.1
            self.scale_spline = 1.0

            self.reset_parameters()

      def reset_parameters(self):
            torch.nn.init.kaiming_uniform_(self.base_weights, a=math.sqrt(5) * self.scale_base)
            with torch.no_grad():
                noise = (
                    (
                        torch.rand(self.grid_size + 1, self.in_channels, self.out_channels)
                        - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
                ).to(self.device)

                for i in range(len(self.spline_weights)):
                    self.spline_weights[i].data.copy_(
                        (self.scale_spline)
                        * self.curve2coeff(
                            self.grid[i].T[self.spline_order : -self.spline_order],
                            noise,
                            i,
                            True
                        )
                    )

      def curve2coeff(self, x: torch.Tensor, y: torch.Tensor, kernel_idx, init=False):
            #print(x.shape)
            A = self.b_splines(x, kernel_idx).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
            #print('A', A.shape)
            B = y.transpose(0, 1)
            if self.adaptive and not init:
                ATA = torch.bmm(A.transpose(1, 2), A)
                ATB = torch.bmm(A.transpose(1, 2), B)
                ATAreg = ATA + self.lambda_value * torch.eye(ATA.size(1), device=self.device)
                solution = torch.linalg.lstsq(ATAreg, ATB).solution

            else:
                solution = torch.linalg.lstsq(A, B).solution
            #A = A + lambda_value * torch.eye(A.size(0), device=self.device)
            #B = B + self.lambda_value * torch.eye(B.size(0), device=self.device)
            #print(solution.shape)
            result = solution.permute(2, 0, 1)
            return result.reshape(self.out_channels, -1).contiguous()

      @torch.no_grad()
      def update_grid(self, x: torch.Tensor, margin=0.01, kernel_idx=0):
            batch = x.size(0)

            splines = self.b_splines(x, kernel_idx)#.unsqueeze(0)  (batch, in, coeff)
            splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
            orig_coeff = self.spline_weights[kernel_idx].view(self.out_channels, self.in_channels, -1) #self.scaled_spline_weight  # (out, in, coeff)
            orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
            unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
            unreduced_spline_output = unreduced_spline_output.permute(
                1, 0, 2
            )  # (batch, in, out)
            x_sorted = torch.sort(x, dim=0)[0]#.view(1, 3)
            grid_adaptive = x_sorted[
                torch.linspace(
                    0, batch-1, self.grid_size + 1, dtype=torch.int64, device=x.device
                )
            ]
            uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
            grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
            )
            new_grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

            new_grid = torch.cat(
                [
                    new_grid[:1]
                    - uniform_step
                    * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                    new_grid,
                    new_grid[-1:]
                    + uniform_step
                    * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
                ],
                dim=0,
            )
            self.grid[kernel_idx].copy_(new_grid.T)
            self.spline_weights[kernel_idx].data.copy_(self.curve2coeff(x, unreduced_spline_output, kernel_idx))

      
      def b_splines(self, x: torch.Tensor, kernel_idx):
            x = x.unsqueeze(-1)
            bases = ((x >= self.grid[kernel_idx][:,:-1]) & (x < self.grid[kernel_idx][:,1:])).to(torch.float32)
            for k in range(1, self.spline_order + 1):
                bases = (
                    (x - self.grid[kernel_idx][:,:-(k+1)])
                    / (self.grid[kernel_idx][:,k:-1] - self.grid[kernel_idx][:,:-(k+1)])
                    * bases[:,:,:-1]
                ) + (
                    (self.grid[kernel_idx][:,k+1:] - x)
                    / (self.grid[kernel_idx][:,k+1:] - self.grid[kernel_idx][:,1:(-k)])
                    * bases[:,:,1:]
                )
            return bases.contiguous()


      def forward(self, x: spconv.SparseConvTensor):
            ## Currently supporting only sparseconv tensors
            #print(f"x features shape: {x.features.shape}")
            ## Calculate input output pairs
            outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(x.indices,
                                                                         x.batch_size,
                                                                         x.spatial_shape,
                                                                         ConvAlgo.Native,
                                                                         self.kernel_size,
                                                                         self.stride,
                                                                         self.padding,
                                                                         self.dilation,
                                                                         self.output_padding,
                                                                         self.subm,
                                                                         self.transposed)
            # print('indice_pairs', indice_pairs.shape)
            # print('indice_pair_num', indice_pair_num.shape)
            # print('indice_pair_num', indice_pair_num)
            # print('grid', self.grid.shape)
            # print('features', x.features.shape)
            # print('splineweights', self.spline_weights.shape)
            # print('baseweights', self.base_weights.shape)
            ## Copy and calculate some sparse tensor attributes
            out_tensor = x.shadow_copy()
            out_spatial_shape = ops.get_conv_output_size(
                    x.spatial_shape, self.kernel_size, self.stride, self.padding, self.dilation)
            indice_dict = x.indice_dict.copy()
            out_features = torch.zeros((outids.size(0), self.out_channels), device=self.device)

            # print('out_features', out_features.shape)
            features = x.features
            ## Do the actual convolution
            if self.cud:
                 assert self.device == torch.device('cuda')
                 threadsperblock = self.num_kernel_elems
                 blockspergrid = 1
                 activated = self.base_activation(features)
                 result = torch.randn(self.num_kernel_elems, self.in_channels, 12).to(self.device)
                 temp = torch.randn(self.in_channels, 12).to(self.device)

                 ip = cuda.as_cuda_array(indice_pairs)
                 ipn = cuda.as_cuda_array(indice_pair_num)
                 gr = cuda.as_cuda_array(self.grid)
                 fe = cuda.as_cuda_array(features)
                 sw = cuda.as_cuda_array(self.spline_weights.detach())
                 bw = cuda.as_cuda_array(self.base_weights.detach())
                 of = cuda.as_cuda_array(out_features.detach())
                 ac = cuda.as_cuda_array(activated)
                 res = cuda.as_cuda_array(result)
                 temp = cuda.as_cuda_array(temp)

                 cuda_tools.test_conv[blockspergrid, threadsperblock](ip, ipn, fe, ac, of, sw, bw, gr, self.grid_size, self.spline_order, self.grid_eps, res, temp)
                 out_features = torch.tensor(of.copy_to_host()).to(self.device)
            else:
                ## Proxy convolution Function
                for kernel_idx in range(self.num_kernel_elems):
                    ### DO THIS IN PARALLEL PER KERNEL ELEMENT ###
                    if (indice_pair_num[kernel_idx] == 0):
                         continue
                    iopairs = indice_pairs[:,kernel_idx,:indice_pair_num[kernel_idx]] # all the input-output pairs for kernel
                    
                    #print(iopairs.shape)
                    inp = iopairs[0, :]
                    out = iopairs[1, :]
                    #print(inp)
                    #print(features[inp].shape)
                    #print(features[inp.long()].shape)
                    x = features[inp.long()]
                    if self.adaptive:
                        self.update_grid(x, margin=0.01, kernel_idx=kernel_idx)
                    bases = self.b_splines(x, kernel_idx)
                    
                    out_features[out.long()] += (
                        F.linear(bases.view(-1, bases.size(-1)*bases.size(-2)), self.spline_weights[kernel_idx]).squeeze(0) +
                        F.linear(self.base_activation(x), self.base_weights[kernel_idx])
                    ).squeeze(0)


            #print("return of layer", type(out_tensor))
            out_tensor = out_tensor.replace_feature(out_features)
            out_tensor.indices = outids
            out_tensor.indice_dict = indice_dict
            out_tensor.spatial_shape = out_spatial_shape
            #print(out_tensor.batch_size)
            #out_tensor.batch_size = x.batch_size
            
            return out_tensor
      
      def __repr__(self):
        # Customize this method to include all relevant details
        return (f'{self.__class__.__name__}('
                f'{self.in_channels}, '
                f'{self.out_channels}, '
                f'kernel_size={self.kernel_size}, '
                f'stride={self.stride}, '
                f'padding={self.padding}, '
                f'dilation={self.dilation}, '
                f'output_padding={self.output_padding}, '
                f'bias={self.bias})')
      
class SubMKANConv3d(SparseKANConv3d):
     def __init__(self,
                   ndim: int,
                   in_channels: int,
                   out_channels: int,
                   kernel_size=3,
                   stride=1,
                   padding=0,
                   dilation=1,
                   groups=1,
                   bias: bool = False,
                   output_padding=0,
                   transposed: bool = False,
                   grid_size=3,
                   spline_order=3,
                   grid_range=[-1, 1],
                   grid_eps=0.02,
                   base_activation = torch.nn.SiLU,
                   device='cpu',
                   use_numba = False,
                   adaptive=False):
            super(SubMKANConv3d, self).__init__(ndim, 
                                                in_channels, 
                                                out_channels, 
                                                kernel_size, 
                                                stride, 
                                                padding, 
                                                dilation, 
                                                groups, 
                                                bias, 
                                                True, 
                                                output_padding, 
                                                transposed, 
                                                grid_size, 
                                                spline_order, 
                                                grid_range, 
                                                grid_eps, 
                                                base_activation,
                                                device, 
                                                use_numba,
                                                adaptive)
      
if __name__ == '__main__':
    # Test SparseKANConv3D
    # Create a SparseConvTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = PointToVoxel(
        vsize_xyz=[0.1, 0.1, 0.1],
        coors_range_xyz=[-80, -80, -2, 80, 80, 6],
        num_point_features=3,
        max_num_voxels=5000,
        max_num_points_per_voxel=5)
    pc = np.random.uniform(-10, 10, size=[1000, 3])
    pc_th = torch.from_numpy(pc)
    voxels, coords, num_points_per_voxel = gen(pc_th, empty_mean=True)

    #print(voxels.shape)
    indices = torch.cat((torch.zeros(voxels.shape[0], 1), coords[:, [2,1,0]]), dim=1).to(torch.int32)
    features = torch.max(voxels, dim=1)[0]
    spatial_shape = [1600, 1600, 80]
    batch_size = 1
    features = features.to(device)
    indices = indices.to(device)
    test_sparse = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
    #print(f"sparse input features shape: {test_sparse.features.shape}")
    #print(f"sparse input indices shape: {test_sparse.indices.shape}")
    #print(test_sparse.indices)
    # Create a SparseKANConv3D
    kan_conv = SparseKANConv3d(3, 3, 5, device=device, subm=False)
    # Perform a forward pass
    out = kan_conv(test_sparse)