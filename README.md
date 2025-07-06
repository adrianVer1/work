# UMFusion论文复现
 

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/wdhudiekou/UMF-CMGR/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.3.1-%237732a8)](https://pytorch.org/)


### Unsupervised Misaligned Infrared and Visible Image Fusion via Cross-Modality Image Generation and Registration [IJCAI2022 Oral Presentation]

By Di Wang, Jinyuan Liu, Xin Fan, and Risheng Liu

<div align=center>
<img src="https://github.com/wdhudiekou/UMF-CMGR/blob/main/Fig/network.png" width="80%">
</div>

原论文地址 （https://github.com/wdhudiekou/UMF-CMGR）
## Previous Requirements
- CUDA 10.1
- Python 3.6 (or later)
- Pytorch 1.6.0
- Torchvision 0.7.0
- OpenCV 3.4
- Kornia 0.5.11

因为Python 3.6 在 2021 年底停止维护（EOL），现在主流的深度学习框架（如 PyTorch 等）都已经不再支持它，所以复现时使用python3.8

## New Requirements
- CUDA: 11.8
- Python: 3.8
- Pytorch: 2.3.1
- Torchvision: 0.18.1
- OpenCV: 4.10.0 (包名为 opencv-python)
- Kornia: 0.7.2

## 下载对应数据集
Please download the following datasets:
*   [RoadScene](https://github.com/hanna-xu/RoadScene)
*   [TNO](http://figshare.com/articles/TNO\_Image\_Fusion\_Dataset/1008029)

## 下载预训练模型
download the [pretrained model](https://pan.baidu.com/s/1JO4hjdaXPUScCI6oFtPEnQ) (code: i9ju) of CPSTN and put it into folder './CPSTN/checkpoints/pretrained/'
download(https://pan.baidu.com/s/199dqOLHyJS9aY5YecuVglA) (code: hk25) of the registration network MRRN.
download the [pretrained model](https://pan.baidu.com/s/1GZrYrg_qzAfQtoCrZLJsSw) (code: 0rbm) of fusion network DIFN.

将MRRO和DIFN的预训练模型放到\UMF-CMGR-main\checkpoints中

## 开始测试
1.使用CPSTN将可见光转为伪红外
 ```
       cd cpstn
       python test.py  --dataroot ../datasets/rgb2ir/RoadScene/trainA  --name rgb2ir_paired_Road_edge_pretrained  --model test  --no_dropout  --preprocess none  --checkpoints_dir ./checkpoints/pretrained models.test_model
       python test.py  --dataroot ../datasets/rgb2ir/RoadScene/testA  --name rgb2ir_paired_Road_edge_pretrained  --model test  --no_dropout  --preprocess none  --checkpoints_dir ./checkpoints/pretrained models.test_model
```
2.使用MRRN配准,对齐失真红外与伪红外
```
       cd ../data
       Python get_test_data.py --ir ../datasets/rgb2ir/RoadScene/trainB --vi ../datasets/rgb2ir/RoadScene/trainA --dst ./deformable/train
       python get_test_data.py --ir ../datasets/rgb2ir/RoadScene/testB  --vi ../datasets/rgb2ir/RoadScene/testA  --dst ./deformable/test
       python get_svs_map.py
       cd test
       Python test_reg.py  \
         --it ../CPSTN/results/rgb2ir_paired_Road_edge_pretrained/test_latest/images  \
         --ir   ../data/deformable/test/ir_warp  \
         --disp ../data/deformable/test  \
         --ckpt ../checkpoints/mrrn/best_model.pth  \
         --dst  ../results/registration \
 ```
配准结果如下
```
UMF-CMGR-main\results\registration
├─ ir/          # 原始（失真）红外
├─ it/          # 伪红外（CPSTN 输出）
├─ ir_reg/      # 配准后的红外
├─ grid/        # 原始网格可视化
├─ warp_grid/   # 变形后网格可视化
├─ reg_grid/    # 配准后网格可视化
├─ ir_warp_grid/ # 变形红外叠加网格
└─ ir_reg_grid/ # 配准红外叠加网格
```
3.使用DIFN融合,把配准后的红外图和原始可见光图进行特征级融合，生成最终的融合图
```
       cd Test
       python test_fuse.py  
         --ir   ../results/registration/ir_reg  
         --vi ../datasets/rgb2ir/RoadScene/testA  
         --ckpt ../checkpoints/DIFN/best_model.pth  
         --dst  ../results/fusion 
         --dim  64
```
融合结果如下
```
UMF-CMGR-main\results\fusion
├─fused/: 最终融合图
├─ir/: 配准前的红外图
└─ vi/: 对应的可见光图
```

4.配准评估
```
       python metrics.py
```















