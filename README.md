# MSG-Transformer

Official implementation of the paper [MSG-Transformer: Exchanging Local Spatial Information by Manipulating Messenger Tokens](https://arxiv.org/abs/2105.15168),  
by [Jiemin Fang](https://jaminfong.cn/), [Lingxi Xie](http://lingxixie.com/), [Xinggang Wang](https://xinggangw.info/), [Xiaopeng Zhang](https://sites.google.com/site/zxphistory/), [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/), [Qi Tian](https://scholar.google.com/citations?hl=en&user=61b6eYkAAAAJ).

We propose a novel Transformer architecture, named MSG-Transformer, which enables efficient and flexible information exchange by introducing MSG tokens to sever as the information hub.

-----------------------------

Transformers have offered a new methodology of designing neural networks for visual recognition. Compared to convolutional networks, Transformers enjoy the ability of referring to global features at each stage, yet the attention module brings higher computational overhead that obstructs the application of Transformers to process high-resolution visual data. This paper aims to alleviate the conflict between efficiency and flexibility, for which we propose a specialized token for each region that serves as a messenger (MSG). Hence, by manipulating these MSG tokens, one can flexibly exchange visual information across regions and the computational complexity is reduced. We then integrate the MSG token into a multi-scale architecture named MSG-Transformer. In standard image classification and object detection, MSG-Transformer achieves competitive performance and the inference on both GPU and CPU is accelerated.
![block](./imgs/block.png)
![arch](./imgs/arch.png)

## Updates
* 2021.6.2 Code for ImageNet classification is released.

## Requirements
* PyTorch==1.7
* timm==0.3.2
* Apex
* opencv-python>=3.4.1.15
* yacs==0.1.8

## Data Preparation
Please organize your ImageNet dataset as followins.
```
path/to/ImageNet
|-train
| |-cls1
| | |-img1
| | |-...
| |-cls2
| | |-img2
| | |-...
| |-...
|-val
  |-cls1
  | |-img1
  | |-...
  |-cls2
  | |-img2
  | |-...
  |-...
```

## Training
Train MSG-Transformers on ImageNet-1k with the following script.  
For `MSG-Transformer-T`, run
```
python -m torch.distributed.launch --nproc_per_node 8 main.py \
    --cfg configs/msg_tiny_p4_win7_224.yaml --data-path <dataset-path> --batch-size 128
```
For `MSG-Transformer-S`, run
```
python -m torch.distributed.launch --nproc_per_node 8 main.py \
    --cfg configs/msg_small_p4_win7_224.yaml --data-path <dataset-path> --batch-size 128
```
For `MSG-Transformer-B`, we recommend running the following script on two nodes, where each node is with 8 GPUs.
```
python -m torch.distributed.launch --nproc_per_node 8 \
    --nnodes=2 --node_rank=<node-rank> --master_addr=<ip-address> --master_port=<port> \
    main.py --cfg configs/msg_base_p4_win7_224.yaml --data-path <dataset-path> --batch-size 64
```

## Evaluation
Run the following script to evaluate the pre-trained model.
```
python -m torch.distributed.launch --nproc_per_node <GPU-number> main.py \
    --cfg <model-config> --data-path <dataset-path> --batch-size <batch-size> \
    --resume <checkpoint> --eval
```

## Main Results
### ImageNet-1K
| **Model** | **Input size** | **Params** | **FLOPs** | **GPU throughput (images/s)** | **CPU Latency** | Top-1 ACC (%) |
|-----------|----------------|------------|------------|------------|------------|------------|
| MSG-Trans-T | 224 | 28M  | 4.6G | 696.7 | 150ms  | 80.9 |
| MSG-Trans-S | 224 | 50M  | 8.9G | 401.0 | 262ms  | 83.0 |
| MSG-Trans-B | 224 | 88M  | 15.8G  | 262.6 | 437ms  | 83.5 |
### MS-COCO
| **Method** | **box mAP** | **mask mAP** | **Params** | **FLOPs** | **FPS** |
|------------|-------------|--------------|------------|-----------|---------|
| MSG-Trans-T  | 50.3        | 43.6         | 86M        | 748G      | 9.4     |
| MSG-Trans-S  | 51.8        | 44.8         | 107M       | 842G      | 7.5     |
| MSG-Trans-B  | 51.9        | 45.0         | 145M       | 990G      | 6.2     |

## Acknowledgements
This repository is based on [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and [timm](https://github.com/rwightman/pytorch-image-models/). Thanks for their contributions to the community.

## Citation
If you find this repository/work helpful in your research, welcome to cite the paper.
```
@article{fang2021msgtransformer,
  title={MSG-Transformer: Exchanging Local Spatial Information by Manipulating Messenger Tokens},
  author={Jiemin Fang and Lingxi Xie and Xinggang Wang and Xiaopeng Zhang and Wenyu Liu and Qi Tian},
  journal={arXiv:2105.15168},
  year={2021}
}
```
