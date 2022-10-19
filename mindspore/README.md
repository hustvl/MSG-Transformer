# MSG-Transformer on MindSpore

##  Installation  

1. create python 3.8 environment  
```bash
conda create -n msg-transformer-ms python=3.8  
```
2. activate the new environment  
```bash
conda activate msg-transformer-ms    
```

3. install mindspore   
``` bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.8.1/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-1.8.1-cp38-cp38-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

4. install indepencies   
```bash
pip install mindvision pycocotools opencv-python numpy yacs scipy  
```

##  Demo

```bash
python test.py --cfg ./configs/msg_tiny_p4_win7_224.yaml    
```
