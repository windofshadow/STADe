from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import inspect
import os
if __name__ == '__main__':
    # 获取当前脚本所在目录（通常是工程目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建目标文件的绝对路径
    path_boundary_max_pooling_cuda = "AFSD/prop_pooling/boundary_max_pooling_cuda.cpp"  # 替换为你的文件相对路径
    path_boundary_max_pooling_kernel="AFSD/prop_pooling/boundary_max_pooling_kernel.cu"
    absolute_path_boundary_max_pooling_cuda = os.path.join(current_dir, path_boundary_max_pooling_cuda)
    absolute_path_boundary_max_pooling_kernel = os.path.join(current_dir,path_boundary_max_pooling_kernel)
    
    setup(
        name='AFSD',
        version='1.0',
        description='Learning Salient Boundary Feature for Anchor-free '
                    'Temporal Action Localization',
        author='Chuming Lin, Chengming Xu',
        author_email='chuminglin@tencent.com, cmxu18@fudan.edu.cn',
        packages=find_packages(
            exclude=('configs', 'models', 'output', 'datasets')
        ),
        ext_modules=[
            CUDAExtension('boundary_max_pooling_cuda', [
                absolute_path_boundary_max_pooling_cuda,
                absolute_path_boundary_max_pooling_kernel
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
