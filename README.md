# Arc-VQVAE

![teaser](assets/rec.png)

## Requirements
A suitable [conda](https://conda.io/) environment named `taming` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate taming
```

`environment.yaml`의 cuda 및 torch 버전은 본인의 gpu 환경에 맞게끔 설정해주시면 됩니다.  
(Please set the cuda and torch versions in environment.yaml to match your GPU environment.)



## Data Preparation

### ImageNet
The code will try to download (through [Academic
Torrents](http://academictorrents.com/)) and prepare ImageNet the first time it
is used. However, since ImageNet is quite large, this requires a lot of disk
space and time. If you already have ImageNet on your disk, you can speed things
up by putting the data into
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/` (which defaults to
`~/.cache/autoencoders/data/ILSVRC2012_{split}/data/`), where `{split}` is one
of `train`/`validation`. It should have the following structure:

```
${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/
├── n01440764
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   ├── ...
├── n01443537
│   ├── n01443537_10007.JPEG
│   ├── n01443537_10014.JPEG
│   ├── ...
├── ...
```

If you haven't extracted the data, you can also place
`ILSVRC2012_img_train.tar`/`ILSVRC2012_img_val.tar` (or symlinks to them) into
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_train/` /
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_validation/`, which will then be
extracted into above structure without downloading it again.  Note that this
will only happen if neither a folder
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/` nor a file
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/.ready` exist. Remove them
if you want to force running the dataset preparation again.


### FFHQ
Create a symlink `data/ffhq` pointing to the `images1024x1024` folder obtained
from the [FFHQ repository](https://github.com/NVlabs/ffhq-dataset).

`taming/data/faceshq.py` 파일에 FFHQTrain, FFHQValidation 의 root 경로를 본인의 FFHQ 데이터셋 경로로 지정해주면 됩니다.  
(Please specify the root path of FFHQTrain and FFHQValidation in the `taming/data/faceshq.py` file as the path to your FFHQ dataset.)



## Training models

### FFHQ

Train a VQGAN with
```
python main.py --base configs/faceshq_vqgan.yaml -t True --gpus "0,"
```

Train a VQGAN on ImageNet with
```
python main.py --base configs/imagenet_vqgan.yaml -t True --gpus "0,"
```

pretrained model의 학습을 재개하고 싶으면 `--resume "your/ckpt/path"` 인자를 추가하면 됩니다.  
(If you want to resume training of a pretrained model, you can add the `--resume "your/ckpt/path"` argument.)


config 파일에 있는 하이퍼 파라미터 세팅은 batch_size와 num_workers 값을 제외한 나머지는 최대한 건들지 않는 것을 추천 드립니다. 코드북 학습이 붕괴될 수 있습니다.  
(We recommend leaving all hyperparameter settings in the config file unchanged, except for batch_size and num_workers, as this can disrupt codebook learning.)

## Arc-VQVAE Implementation

ArcLoss는 `taming/modules/losses/vqperceptual.py` 파일에 구현되어 있으며,  
BBNR은 `taming/modules/vqvae/quantize.py` 파일에 구현되어 있으니 참조 바랍니다. 

(ArcLoss is implemented in the `taming/modules/losses/vqperceptual.py` file,  
and BBNR is implemented in the `taming/modules/vqvae/quantize.py` file.)