# Implementation of COPIA: Counterfactual-Guided Diversified Path Integration Neuron Attribution Attack

## Prerequisites

The implementation requires the following software and libraries:

- Python 3.6.13
- Keras 2.2.4
- NumPy 1.16.2
- TensorFlow 1.14.0
- TQDM 4.63.1
- Pillow 6.0.0
- SciPy 1.2.1

## Pre-trained Models

For our experiments, we rely on publicly available pre-trained models.
The following table summarizes the required models along with their corresponding download links.


Model  | Source
------------- | -------------
Inception v3  | [Download](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)
Inception v4| [Download](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz)
Inception-ResNet-v2  | [Download](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz)
ResNet v2 152  | [Download](http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz)
Inception v3 adv | [Download](http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz)
Inception ResNet v2 adv  | [Download](http://download.tensorflow.org/models/adv_inception_resnet_v2_2017_12_18.tar.gz)
Inception v3 adv ens3  | [Download](http://download.tensorflow.org/models/ens3_adv_inception_v3_2017_08_18.tar.gz)
Inception v3 adv ens4  | [Download](http://download.tensorflow.org/models/ens4_adv_inception_v3_2017_08_18.tar.gz)
Inception ResNet v2 adv ens3  | [Download](http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz)|


The standard (non-adversarial) models are obtained from [TensorFlow's Slim Model Library](https://github.com/tensorflow/models/tree/master/research/slim), whereas the adversarially trained models are downloaded from the [Adversarial ImageNet Models Repository](https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models). After downloading, all model files should be placed in the `models` directory of the project repository.


## Execution Instructions

### Main Python Files

- `COPIA_5_xr_grad_hybrid_inception_resnet_v2.py` : implements the COPIA attack using Inception-ResNet-v2 as the surrogate model.
This version employs five counterfactual examples, with diversified path updates that combine both gradient ascent and gradient descent.

- `verify.py` : provides the evaluation pipeline for assessing the generated adversarial examples across various target models.


### Running COPIA Attacks

To execute the COPIA attack method, use the following command format, adjusting arguments as needed for the chosen surrogate model and experimental configuration.

For instance, to run COPIA on the Inception-ResNet-v2 model, use the following command:
```bash
python COPIA_5_xr_grad_hybrid_inception_resnet_v2.py --model_name inception_resnet_v2 --attack_method Ours --layer_name InceptionResnetV2/InceptionResnetV2/Conv2d_2b_3x3/Relu --ens 30 --output_dir ./COPIA_adv/inception_resnet_v2_xr_5_lr_0.025_scale_0.2_grad_hybrid_COPIA/  --batch_size 2 --lr 0.025 --scale 0.2 --image_size 299 --image_resize 331 
```

To run COPIA on the Inception-ResNet-v2 model with PIDI enabled, use the following command:


```bash
python COPIA_5_xr_grad_hybrid_inception_resnet_v2.py --model_name inception_resnet_v2 --attack_method OursPIDI --layer_name InceptionResnetV2/InceptionResnetV2/Conv2d_2b_3x3/Relu --ens 30 --output_dir ./COPIA_PIDI_adv/inception_resnet_v2_xr_5_lr_0.025_scale_0.2_grad_hybrid_COPIA_PIDI/  --batch_size 2 --lr 0.025 --scale 0.2 --image_size 299 --image_resize 331 
```

### Verification

To verify the effectiveness of the generated adversarial examples, use the `verify.py` script as shown below:

```bash
python verify.py --ori_path ./dataset/images/ --adv_path ./COPIA_adv/inception_resnet_v2_xr_5_lr_0.025_scale_0.2_grad_hybrid_COPIA/
```

### For Other Source Models

- COPIA_5_xr_grad_hybrid_inception_v3.py: implementation for inception v3
- COPIA_5_xr_grad_hybrid_inception_v4.py: implementation for inception v4
- COPIA_5_xr_grad_hybrid_resnet_v2_50.py: implementation for resnet_v2_50
- COPIA_5_xr_grad_hybrid_resnet_v2_152.py: implementation for resnet_v2_152

## Hardware:
The experiments were conducted on an RTX 3090 GPU with 24 GB of memory.


## Refenece
Code refer to: [NAA](https://github.com/jpzhang1810/NAA),[DANAA](https://github.com/Davidjinzb/DANAA), [MFAA](https://github.com/KWPCCC/MFAA), [IG2](https://github.com/JoeZhuo-ZY/IG2).
