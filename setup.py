from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Ensure we can find the source files
src_dir = os.path.join("src", "tic_tac_toe_bolt", "mcts_cpp")
sources = [os.path.join(src_dir, "mcts.cpp")]

setup(
    name='tic_tac_toe_bolt_cpp',
    ext_modules=[
        CppExtension(
            name='tic_tac_toe_bolt._mcts_cpp',
            sources=[os.path.join("src", "tic_tac_toe_bolt", "mcts_cpp", "mcts.cpp")],
            extra_compile_args=['-std=c++17']
        ),
        CppExtension(
            name='cublino_contra._mcts_cpp',
            sources=[os.path.join("src", "cublino_contra", "mcts_cpp", "mcts.cpp")],
            extra_compile_args=['-std=c++17']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    package_dir={'': 'src'}
)
