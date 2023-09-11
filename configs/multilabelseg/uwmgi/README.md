# Multi-label Semantic Segmentation with UWMGI Dataset 

### Preparation
* Install PaddlePaddle and relative environments based on the [installation guide](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html).
* Install PaddleSeg based on the [reference](../../../docs/install.md).
* Download the UWMGI dataset and link to PaddleSeg/data. 

```shell
wget https://bj.bcebos.com/v1/ai-studio-online/01c05a320b8745ca9985a7e62f12286a8b6f985a36e643eca1fb2b6f29fa21c7?responseContentDisposition=attachment%3B%20filename%3DUWMGI.zip&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2023-09-10T11%3A24%3A21Z%2F-1%2F%2F37523e8456373c755318dee5d7a109f30681185f60ebef593cffbec8f3cf23f6
mkdir -p data/
unzip UWMGI.zip -d data/
```

```
PaddleSeg/data
├── UWMGI
│   ├── images
│   │   ├── ......
│   ├── annotations
│   │   ├── train.json
│   │   ├── val.json
│   │   ├── ......
```

### Training
You can start training by assign the ```tools/train.py``` with config files, the config files are under ```PaddleSeg/configs/multilabelseg/uwmgi```. Details about training are under [training guide](../../../docs/train/train.md). You can find the trained models under ```Paddleseg/save_dir/best_model/model.pdparams```

```bash
# multi-gpu
export CUDA_VISIBLE_DEVICES=0,1
python3 -m paddle.distributed.launch tools/train.py \
    --config configs/multilabelseg/uwmgi/pp_mobileseg_tiny_uwmgi_512x512_80k.yml \
    --save_dir output/pp_mobileseg_tiny_uwmgi_512x512_80k \
    --save_interval 1000 \
    --num_workers 8 \
    --log_iters 100 \
    --use_ema \
    --do_eval \
    --use_vdl

# single-gpu
python3 -m tools/train.py \
    --config configs/multilabelseg/uwmgi/pp_mobileseg_tiny_uwmgi_512x512_80k.yml \
    --save_dir output/pp_mobileseg_tiny_uwmgi_512x512_80k \
    --save_interval 1000 \
    --num_workers 8 \
    --log_iters 100 \
    --use_ema \
    --do_eval \
    --use_vdl
```

### Validation
With the trained model on hand, you can verify the model's accuracy through evaluation. Details about evaluation are under [evaluation guide](../../../docs/evaluation/evaluate.md).

```bash
# multi-gpu
export CUDA_VISIBLE_DEVICES=0,1
python3 -m paddle.distributed.launch tools/val.py \
    --config configs/multilabelseg/uwmgi/pp_mobileseg_tiny_uwmgi_512x512_80k.yml \
    --model_path output/pp_mobileseg_tiny_uwmgi_512x512_80k/best_model/model.pdparams
    
# single-gpu
python3 tools/val.py \
    --config configs/multilabelseg/uwmgi/pp_mobileseg_tiny_uwmgi_512x512_80k.yml \
    --model_path output/pp_mobileseg_tiny_uwmgi_512x512_80k/best_model/model.pdparams
```

### Inference
With the trained model on hand, you can use the model to predict and visualize images. Details about inference are under [inference guide](../../../docs/predict/predict.md).

```bash
python tools/predict.py \
       --config configs/multilabelseg/uwmgi/pp_mobileseg_tiny_uwmgi_512x512_80k.yml \
       --model_path output/pp_mobileseg_tiny_uwmgi_512x512_80k/best_model/model.pdparams \
       --image_path data/UWMGI/images/case147_day14_slice_0095.png \
       --save_dir output/result \
       --use_multilabel
```