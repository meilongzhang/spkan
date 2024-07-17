import torch.nn as nn
import spconv.pytorch as spconv


if __name__ == "__main__" and __package__ is None:
    import os
    import sys
    currentdir = os.path.dirname(os.path.abspath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    import conv as kanv
else:
    print('else')


class SECOND_NECK(nn.Module):
    def __init__(self):
        super(SECOND_NECK, self).__init__()
        
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
            kernel_size=3,
            stride=2,
            padding=0,
            dilation=1,
            output_padding=0,
            use_numba=False
        )

    def _block(self, manifold, in_channels, out_channels, kernel_size, stride, padding, dilation, output_padding, use_numba):
        if manifold:
            return spconv.SparseSequential(
                kanv.SubMKANConv3d(ndim=3, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, output_padding=output_padding, use_numba=use_numba),
                nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            )
        else:
            return spconv.SparseSequential(
                kanv.SparseKANConv3d(ndim=3, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, output_padding=output_padding, use_numba=use_numba),
                nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            )

if __name__ == '__main__':
    model = SECOND_NECK()
    print(model)