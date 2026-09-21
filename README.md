# MetaphorCBM

**Autonomous Concept Bottleneck Construction via Multimodal Sparse Dictionary Fusion**

MetaphorCBM autonomously constructs concept bottlenecks by fusing visual and textual sparse dictionaries learned from target domain data. Its structured coupling uses optimal transport to alternately optimize sample conditioned local correspondences and dictionary level global correspondences, forming multimodal concepts for interpretation and prediction.

This repository provides implementations of sparse dictionary learning, structured multimodal coupling, downstream classification, and concept completeness evaluation, together with experiment configurations, fixed data splits, and result records.

## Installation

Use Python 3.11 with PyTorch 2.7.0 and torchvision 0.22.0. Install the PyTorch build appropriate for your system using the [official installation instructions](https://pytorch.org/get-started/previous-versions/#v270). For CUDA 12.8:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The supplied configurations use the pretrained RN50 visual backbone from [OpenAI CLIP](https://github.com/openai/CLIP) and the `bert-base-uncased` tokenizer for text preprocessing. These resources are downloaded on first use.

Run all commands below from the repository root. The supplied experiment configurations use a CUDA GPU.

## Repository structure

| Directory | Contents |
| --- | --- |
| `metaphorcbm/` | Models, data utilities, training, coupling, and downstream evaluation |
| `scripts/` | Command line entry points |
| `configs/` | Dictionary learning, coupling, classification, and completeness configurations |
| `results/` | Reported results and fixed dataset splits |

Configurations specify data, checkpoint, and output paths relative to the repository root. Set these paths to match your local files. Generated checkpoints, feature caches, and evaluation outputs are written to `outputs/`.

## Data preparation

### MS-COCO

Download the 2017 training images, validation images, and train/validation annotations from [MS-COCO](https://cocodataset.org/#download). Arrange the files as follows:

```text
data/coco/
  train2017/
  val2017/
  annotations/
    captions_train2017.json
    captions_val2017.json
    instances_train2017.json
    instances_val2017.json
```

Dictionary learning and coupling use image captions. The COCO completeness evaluation uses category labels from the instance annotation files.

### ImageNet

Obtain ImageNet-1K from [ImageNet](https://www.image-net.org/download.php). Organize both training and validation images into class directories:

```text
data/imagenet/
  train/
    n01440764/
    ...
  val/
    n01440764/
    ...
```

Use matching class directory names in `train/` and `val/`.

### CUB-200-2011

Prepare the [CUB-200-2011 images](https://www.vision.caltech.edu/datasets/cub_200_2011/) and [Reed et al. captions](https://github.com/reedscot/cvpr2016) with:

```bash
python -m scripts.prepare_cub200 --data_root data
```

The script downloads and organizes the data under `data/cub200/`. It creates `train/` and `test/` for classification, together with `train2017/`, `val2017/`, and COCO format caption annotations for sparse autoencoder (SAE) training.

### CIFAR and Tiny ImageNet

The dataset loaders download [CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) and Tiny ImageNet into `data/` when needed.

## Sparse dictionary learning

### Text backbone pretraining

Pretrain the text backbone on the captions for the corresponding dataset:

```bash
python -m scripts.pretrain_text_backbone --data_root data/coco --output_dir outputs/text_backbone/coco_rn50
python -m scripts.pretrain_text_backbone --data_root data/cub200 --output_dir outputs/text_backbone/cub200_rn50
```

Each command saves `text_backbone_pretrained.pt` in its output directory. Set `model.text_backbone_weights` in the corresponding SAE configuration to the generated file.

### SAE training

Train the visual and textual sparse dictionaries on COCO or CUB, or train the visual dictionary on ImageNet. Example commands:

```bash
python -m scripts.train_joint_sae --config configs/sae/coco_rn50.json
python -m scripts.train_imagenet_image_sae --config configs/sae/imagenet_rn50.json
```

The [configuration index](configs/README.md#sparse-autoencoders) lists commands for all datasets. SAE checkpoints are saved as `checkpoints/best_model.pth` under each experiment's output directory. The coupling and evaluation configurations specify the checkpoint to load.

## Multimodal coupling

Learn correspondences between visual and textual concept bases on COCO:

```bash
python -m scripts.run_coupling --config configs/coupling/coco_rn50.json
```

The pipeline loads the visual SAE, text backbone, and textual SAE from the joint checkpoint. It filters concept bases and alternately optimizes local and global couplings using unbalanced Gromov–Wasserstein optimal transport.

The output directory contains the coupling matrix (`coupling.npz`), concept statistics (`concept_stats.npz`), and run metadata (`metadata.json`).

## Downstream classification

Train classifiers on spatial concept features for CIFAR-10, CIFAR-100, CUB-200-2011, and ImageNet. For example, on CIFAR-10:

```bash
python -m scripts.run_cifar_downstream --config configs/downstream/cifar10_rn50.json
```

Each configuration runs feature preparation, model selection, and final training through `stage=all`. The individual stages are also available through `--stage prepare`, `--stage select`, and `--stage final`.

See the [configuration index](configs/README.md#downstream-classification) for all dataset commands.

## Concept completeness

Evaluate how well the concept bottleneck preserves predictive information by comparing a linear classifier on concept features with a linear probe on the backbone representation. For example, on CIFAR-10:

```bash
python -m scripts.run_completeness --config configs/completeness/cifar10_rn50.json
```

See the [configuration index](configs/README.md#completeness-evaluation) for experiments on CIFAR-10, CIFAR-100, CUB-200-2011, Tiny ImageNet, and MS-COCO. These evaluations use the COCO SAE checkpoint, with training budgets and random seeds specified in each configuration.

Render the completeness figure from the result records in `results/completeness/`:

```bash
python -m scripts.render_completeness_figure --output-dir outputs/figures
```

The command creates `completeness_fewshot.pdf` and a PNG preview.

Reported results and links to their configurations and detailed records are collected in [results/paper_results.json](results/paper_results.json).

## Citation

If you use MetaphorCBM in your research, please cite:

```bibtex
@unpublished{zhu2026metaphorcbm,
  title = {{MetaphorCBM}: Autonomous Concept Bottleneck Construction via Multimodal Sparse Dictionary Fusion},
  author = {Zhu, Boxuan and Yuan, Qiao and Huang, Weizhi and Gairing, Martin and Pan, Yushan and Guan, Sheng-Uei and Wang, Wei},
  year = {2026},
  note = {Manuscript},
  url = {https://github.com/JulienInWired/metaphorCBM-code}
}
```

## License

This project is released under the MIT License. See `LICENSE` for the full text.

Copyright (c) 2026 Boxuan Zhu.
