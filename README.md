# Pixel Difference Convolution

This repository contains the PyTorch implementation for 
"Pixel Difference Networks for Efficient Edge Detection" 
by 
[Zhuo Su](https://zhuogege1943.com/homepage/)\*, 
[Wenzhe Liu]()\*, 
[Zitong Yu](https://www.oulu.fi/university/researcher/zitong-yu),
[Dewen Hu](https://dblp.org/pers/h/Hu:Dewen.html), 
[Qing Liao](http://liaoqing.me/),
[Qi Tian](https://scholar.google.com/citations?user=61b6eYkAAAAJ&hl=en),
[Matti Pietikäinen](https://en.wikipedia.org/wiki/Matti_Pietik%C3%A4inen_(academic)) and 
[Li Liu](http://lilyliliu.com/)\*\* 
(\* Authors have equal contributions, \*\* Corresponding author). \[[arXiv](https://arxiv.org/abs/2108.07009), [youtube](https://www.youtube.com/watch?v=jEAh_4wm1UU)\]

Please find the associated PidiNet Paper on link [our paper](pdc.bib). 

```
@ARTICLE{pdc-PAMI2023,
  author={Su, Zhuo and Zhang, Jiehua and Wang, Longguang and Zhang, Hua and Liu, Zhen and Pietikäinen, Matti and Liu, Li},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Lightweight Pixel Difference Networks for Efficient Visual Representation Learning}, 
  year={2023},
  volume={45},
  number={12},
  pages={14956-14974},
  keywords={Image edge detection;Feature extraction;Computational modeling;Biological system modeling;Task analysis;Quantization (signal);Object recognition;Binary neural networks;convolutional neural networks;edge detection;efficient representation learning;facial recognition;image classification},
  doi={10.1109/TPAMI.2023.3300513}
}
```

## Running environment

Training: Pytorch __ with cuda __ and cudnn __ with RTX4050 GPU on Lenovo LOQ system <br>
Evaluation: Python Code Instead of Authors MATLAB

*Ealier versions may also work~ :)*

## Dataset

We use the links in [RCF Repository](https://github.com/yun-liu/rcf#testing-rcf) (really thanks for that). The augmented BSDS500, PASCAL VOC, and NYUD datasets can be downloaded with:

```bash
wget http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz
wget http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz
wget http://mftp.mmcheng.net/liuyun/rcf/data/NYUD.tar.gz
```

## Generating edge maps for your own images
```bash
python main.py --model pidinet_converted --config carv4 --sa --dil -j 4 --gpu 0 --savedir /path/to/savedir --datadir /path/to/custom_images --dataset Custom --evaluate /path/to/table5_pidinet/save_models/checkpointxxx.pth --evaluate-converted
```

<div align=center>
<img src="https://user-images.githubusercontent.com/18327074/129970337-bb467a8c-825e-47ee-872c-533f0a5da37a.jpg"><br>
The results of our model look like this. The top image is the messy office table, the bottom image is the peaceful Saimaa lake in southeast of Finland.
</div>

