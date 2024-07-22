import torch.nn as nn
import spconv.pytorch as spconv
from torch import Tensor

try:
    from .. import conv as kanv
except:
    import os
    import sys
    currentdir = os.path.dirname(os.path.abspath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    import conv as kanv


class SECOND_NECK(nn.Module):
    def __init__(self, device):   
        super(SECOND_NECK, self).__init__()   
        self.device = device  
        self.conv_input = self._block(manifold=True, 
                                      in_channels=4, 
                                      out_channels=16, 
                                      kernel_size=3, 
                                      stride=1, 
                                      padding=1, 
                                      dilation=1, 
                                      output_padding=0, 
                                      use_numba=False)

        encoder_layer1 = spconv.SparseSequential(
            self._block(
                manifold=True,
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                output_padding=0,
                use_numba=False
            )
        )

        encoder_layer2 = spconv.SparseSequential(
            self._block(
                manifold=False,
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                output_padding=0,
                use_numba=False
            ),
            self._block(
                manifold=True,
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                output_padding=0,
                use_numba=False
            ),
            self._block(
                manifold=True,
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                output_padding=0,
                use_numba=False
            )
        )

        encoder_layer3 = spconv.SparseSequential(
            self._block(
                manifold=False,
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                output_padding=0,
                use_numba=False
            ),
            self._block(
                manifold=True,
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                output_padding=0,
                use_numba=False
            ),
            self._block(
                manifold=True,
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                output_padding=0,
                use_numba=False
            )
        )

        encoder_layer4 = spconv.SparseSequential(
            self._block(
                manifold=False,
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=[0,1,1],
                dilation=1,
                output_padding=0,
                use_numba=False
            ),
            self._block(
                manifold=True,
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                output_padding=0,
                use_numba=False
            ),
            self._block(
                manifold=True,
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                output_padding=0,
                use_numba=False
            )
        )


        self.encoder_layers = spconv.SparseSequential(
            encoder_layer1,
            encoder_layer2,
            encoder_layer3,
            encoder_layer4
        )

        self.conv_output = self._block(
            manifold=False,
            in_channels=64,
            out_channels=128,
            kernel_size=[3,1,1],
            stride=[2,1,1],
            padding=0,
            dilation=1,
            output_padding=0,
            use_numba=False
        )


    def _block(self, manifold, in_channels, out_channels, kernel_size, stride, padding, dilation, output_padding, use_numba):
        if manifold:
            return spconv.SparseSequential(
                kanv.SubMKANConv3d(ndim=3, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, output_padding=output_padding, use_numba=use_numba, device=self.device),
                nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            )
        else:
            return spconv.SparseSequential(
                kanv.SparseKANConv3d(ndim=3, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, output_padding=output_padding, use_numba=use_numba, device=self.device),
                nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, voxel_features: Tensor, coors: Tensor,
                batch_size: int):
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        if self.return_middle_feats:
            return spatial_features, encode_features
        else:
            return spatial_features

if __name__ == '__main__':
    model = SECOND_NECK()
    print(model)