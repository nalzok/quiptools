from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="quiptools_cuda",
    ext_modules=[
        CUDAExtension(
            name="quiptools",
            sources=["quiptools/quiptools_wrapper.cpp",
                     "quiptools/quiptools.cu",
                     "quiptools/quiptools_e8p_gemv.cu"],
            extra_compile_args={
                "cxx": ["-g", "-lineinfo"],
                "nvcc": ["-O2", "-g", "-Xcompiler", "-rdynamic", "-lineinfo"]
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
