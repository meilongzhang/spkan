from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='basis_extension',
    ext_modules=[
        CUDAExtension(
            name='basis_extension', 
            sources=[
                'bsplines_g.cpp',
                'bsplines.cu',
            ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

