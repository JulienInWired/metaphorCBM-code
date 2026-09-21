# Configuration index

Run these commands from the repository root.

## Sparse autoencoders

```bash
python -m scripts.train_joint_sae --config configs/sae/coco_rn50.json
python -m scripts.train_joint_sae --config configs/sae/cub200_rn50.json
python -m scripts.train_imagenet_image_sae --config configs/sae/imagenet_rn50.json
```

## Cross-modal coupling

```bash
python -m scripts.run_coupling --config configs/coupling/coco_rn50.json
```

## Downstream classification

```bash
python -m scripts.run_cifar_downstream --config configs/downstream/cifar10_rn50.json
python -m scripts.run_cifar_downstream --config configs/downstream/cifar100_rn50.json
python -m scripts.run_cub_downstream --config configs/downstream/cub200_rn50.json
python -m scripts.run_imagenet_downstream --config configs/downstream/imagenet_rn50.json
```

## Completeness evaluation

```bash
python -m scripts.run_completeness --config configs/completeness/cifar10_rn50.json
python -m scripts.run_completeness --config configs/completeness/cifar100_rn50.json
python -m scripts.run_completeness --config configs/completeness/tiny_imagenet_rn50.json
python -m scripts.run_completeness --config configs/completeness/cub200_rn50.json
python -m scripts.run_completeness --config configs/completeness/coco_rn50.json
```
