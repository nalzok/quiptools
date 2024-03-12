from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="quiptools_cuda",
    ext_modules=[
        CUDAExtension(
            name="quiptools",
            sources=["csrc/quiptools/quiptools_wrapper.cpp",
                     "csrc/quiptools/quiptools.cu",
                     "csrc/quiptools/quiptools_e8p_gemv.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"]
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
