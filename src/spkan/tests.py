import unittest
import torch
from spconv.pytorch.utils import PointToVoxel
import numpy as np
import spconv.pytorch as spconv
from conv import SparseKANConv3D
import random

def reset_random(seed_number):
        random.seed(seed_number)
        np.random.seed(seed_number)
        torch.manual_seed(seed_number)
        torch.cuda.manual_seed(seed_number)
        torch.cuda.manual_seed_all(seed_number)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class CudaTest(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.reset_input_data(0)
        self.test_input = spconv.SparseConvTensor(self.features, self.indices, self.spatial_shape, self.batch_size)
        """
        reset_random(0)
        self.kan_conv = SparseKANConv3D(3, 3, 5, device=self.device, use_numba=False)
        reset_random(0)
        self.kan_numba = SparseKANConv3D(3, 3, 5, device=self.device, use_numba=True)
        """

    def reset_input_data(self, num):
        reset_random(num)
        gen = PointToVoxel(
            vsize_xyz=[0.1, 0.1, 0.1],
            coors_range_xyz=[-80, -80, -2, 80, 80, 6],
            num_point_features=3,
            max_num_voxels=5000,
            max_num_points_per_voxel=5)
        pc = np.random.uniform(-10, 10, size=[1000, 3])
        pc_th = torch.from_numpy(pc)
        voxels, coords, num_points_per_voxel = gen(pc_th, empty_mean=True)

        indices = torch.cat((torch.zeros(voxels.shape[0], 1), coords[:, [2,1,0]]), dim=1).to(torch.int32)
        features = torch.max(voxels, dim=1)[0]
        self.spatial_shape = [1600, 1600, 80]
        self.batch_size = 1
        self.features = features.to(self.device)
        self.indices = indices.to(self.device)

    def reset_weights(self, num):
        reset_random(num)
        self.kan_conv = SparseKANConv3D(3, 3, 5, device=self.device, use_numba=False)
        reset_random(num)
        self.kan_numba = SparseKANConv3D(3, 3, 5, device=self.device, use_numba=True)

    def test_bsplines(self):
        for i in range(5):
            self.reset_weights(i)

            cout = self.kan_conv(self.test_input)
            nout = self.kan_numba(self.test_input)

            for i in range(len(cout.features)):
                for j in range(len(cout.features[0])):
                    self.assertAlmostEqual(cout.features[i][j].item(), nout.features[i][j].item(), 1, f'features are not equal: {cout.features[i][j].item()}, {nout.features[i][j].item()}')
                for j in range(len(cout.indices[0])):
                    self.assertEqual(cout.indices[i][j].item(), nout.indices[i][j].item(), f'indices are not equal: {cout.indices[i][j].item()}, {nout.indices[i][j].item()}')

            assert(cout.spatial_shape == nout.spatial_shape)
            assert(cout.batch_size == nout.batch_size)

    def test_kanv(self):
        self.kan_conv = SparseKANConv3D(3, 3, 7, kernel_size=4, stride=2, device=self.device, use_numba=False)
        cout = self.kan_conv(self.test_input)
        assert(cout.features.shape[1] == 7)


if __name__ == '__main__':
    unittest.main()
    print('Tests complete.')