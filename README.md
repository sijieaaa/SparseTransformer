# SpTr: PyTorch Spatially Sparse Transformer Library

Fix some bugs in the original sptr.
This repo supports:
```
torch >= 2.0
torch_geometric >= 2.4.0
```

## Installation
### Install Dependency
torch_geometric >= 2.4.0
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```
or 
```
pip install torch_scatter==2.0.9
pip install torch_geometric==1.7.2
pip install torch_cluster
pip install torch_sparse
```

### Compile sptr
```
python3 setup.py install
```


## Usage
SpTr can be easily used in most current transformer-based 3D point cloud networks, with only several minor modifications. First, define the attention module `sptr.VarLengthMultiheadSA`. Then, wrap the input features and indices into `sptr.SparseTrTensor`, and forward it into the module. That's all. A simple example is as follows. For more complex usage, you can refer to the code of above works (e.g., SphereFormer, StratifiedFormer).
### Example
```
import torch
import numpy as np
import sptr

# Define module
dim = 48

num_heads = 3
indice_key = 'sptr_0'
window_size = np.array([4, 4, 4])  # can also be integers for voxel-based methods
shift_win = False  # whether to adopt shifted window
attn = sptr.VarLengthMultiheadSA(
    dim, 
    num_heads, 
    indice_key, 
    window_size, 
    shift_win
).cuda()

# Wrap the input features and indices into SparseTrTensor. Note: indices can be either intergers for voxel-based methods or floats (i.e., xyz) for point-based methods
# feats: [N, C], indices: [N, 4] with batch indices in the 0-th column
feats = torch.rand(100, dim).cuda()
xyzs = torch.randint(0, 100, (100, 3)).cuda()
indices = torch.cat((torch.zeros_like(xyzs[:, :1]), xyzs), dim=1).cuda()
input_tensor = sptr.SparseTrTensor(feats, indices, spatial_shape=None, batch_size=None)
output_tensor = attn(input_tensor)

# Extract features from output tensor
output_feats = output_tensor.query_feats
```

## Authors

Xin Lai (a Ph.D student at CSE CUHK, xinlai@cse.cuhk.edu.hk) - Initial CUDA implementation, maintainance.

Fanbin Lu (a Ph.D student at CSE CUHK) - Improve CUDA implementation, maintainance.

Yukang Chen (a Ph.D student at CSE CUHK) - Maintainance. 


## Cite

If you find this project useful, please consider citing
```
@inproceedings{lai2023spherical,
  title={Spherical Transformer for LiDAR-based 3D Recognition},
  author={Lai, Xin and Chen, Yukang and Lu, Fanbin and Liu, Jianhui and Jia, Jiaya},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
```
@inproceedings{lai2022stratified,
  title={Stratified transformer for 3d point cloud segmentation},
  author={Lai, Xin and Liu, Jianhui and Jiang, Li and Wang, Liwei and Zhao, Hengshuang and Liu, Shu and Qi, Xiaojuan and Jia, Jiaya},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8500--8509},
  year={2022}
}
```

## License

This project is licensed under the Apache license 2.0 License.
