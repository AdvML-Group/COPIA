#

The official implementation of [**ACM MM 2026**] "Counterfactual-Guided Diversified Path Integration for Transferable Neuron Attribution Attacks". 
This is the implementation of our method COPIA in PyTorch, using a CNN model as the surrogate model.
## Introduction
Transferable adversarial attacks pose significant security threats to
neural networks due to their strong black-box effectiveness against
real-world models. While neuron attribution-based attacks have
shown promise in enhancing transferability, existing methods are
often constrained by integration paths that accumulate noise or
lack structural diversity. Moreover, they typically rely on single layer neuron attribution, failing to exploit multi-layer information.
To address these limitations, we propose COunterfactual-Guided
Diversified Path Integration Neuron Attribution Attack, termed
COPIA. Specifically, we incorporate counterfactual-guided paths
into the neuron attribution framework, to effectively mitigate noise
accumulation in integration trajectories and enable model-adaptive
construction. Building on this, we diversify path update mechanisms and counterfactual samples to enhance path diversity and
improve adversarial transferability. Furthermore, to capture the
complementary semantics across different layers, we design a Cross-layer Attribution (CA) mechanism that progressively aggregates attribution information from deeper layers into the target attack layer
via attribution flow. Extensive experiments on the ImageNet dataset
across normally trained models, adversarially trained CNNs, ViTs,
and various defense models demonstrate that COPIA consistently
outperforms state-of-the-art methods in black-box attack settings.
In addition, the proposed CA mechanism is highly generalizable and
effectively improves the black-box transferability of existing single layer neuron attribution attacks. Code and supplementary materials
are available at https://github.com/AdvML-Group/COPIA.



# Getting Started

## Dependencies

- Python 3.6.13
- Pytorch 1.10.2+cu113
- Torchvision 0.8.2
- Numpy 1.19.5
- pandas 1.1.5 
- Pillow 8.4.0
- Timm 0.5.4
- six 1.17.0
- setuptools 58.0.4 
- Scipy 1.5.4
- tqdm 4.64.1 

## Usage Instructions

##### 1. Models and Datasets

CNN models are all available in [timm](https://github.com/huggingface/pytorch-image-models) and torchvision library. 

The dataset can be downloaded from the link https://github.com/jpzhang1810/NAA, which is provided by NAA's authors, and all the image files should be placed in the directory `. /NAA_images`.

To verify the transferability of adversarial examples, please use the verify.py script in the [NAA](https://github.com/jpzhang1810/NAA) code repository.

##### 2. Source Code
- 'methods/COPIA_Inception_v3_one_cf.py':This is the code for our implementation of COPIA with Inception_v3 as the surrogate model with one counterfactual sample.
- 'methods/COPIA_Inception_v3_five_cf.py':This is the code for our implementation of COPIA with Inception_v3 as the surrogate model with five counterfactual sample.
- 'methods/COPIA_Inception_v4_five_cf.py':This is the code for our implementation of COPIA with Inception_v4 as the surrogate model with five counterfactual sample.
- 'methods/COPIA_Inception_resnet_v2_five_cf.py':This is the code for our implementation of COPIA with Inception_resnet_v2 as the surrogate model with five counterfactual sample.
- 'methods/COPIA_Resnet_50_five_cf.py':This is the code for our implementation of COPIA with Resnet_50 as the surrogate model with five counterfactual sample.
- -'methods/COPIA_Resnet_152_five_cf.py':This is the code for our implementation of COPIA with Resnet_152 as the surrogate model with five counterfactual sample.


##### 3. Quickstart

- You also can use the following command to get all our best results directly!

```
bash run_copia_models.sh
bash run_copia_resnet_models.sh
```

##### 4. Training

- Generate adversarial examples via the proposed COPIA attack on CNN models. You can also modify the hyperparameters to match the experimental settings in our paper.

```
python COPIA_attack.py --attack COPIA_Inception_v3_five_cf --source_model inception_v3
python COPIA_attack.py --attack COPIA_Inception_v4_five_cf --source_model inception_v4
python COPIA_attack.py --attack COPIA_Inception_resnet_v2_five_cf --source_model inception_resnet_v2
python COPIA_attack_resnet.py --attack COPIA_Resnet_50_five_cf --source_model resnet50
python COPIA_attack_resnet.py --attack COPIA_Resnet_152_five_cf --source_model resnet152
```



## Acknowledgments

Code refer to: [NAA](https://github.com/jpzhang1810/NAA),[DANAA](https://github.com/Davidjinzb/DANAA), [MFAA](https://github.com/KWPCCC/MFAA), [IG2](https://github.com/JoeZhuo-ZY/IG2).

We thanks the authors for sharing sincerely.

## Citation
If you find this work is useful in your research, please cite our paper:
```

```

