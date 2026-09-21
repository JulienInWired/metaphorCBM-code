"""Train image and text SAEs with gradient accumulation and validation."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import time
from tqdm import tqdm
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

from ..data import COCODataset, create_coco_dataloaders, ImageTransforms, TextTransforms
from ..models import ResNetWithHooks, create_clip_resnet_backbone, ImageSAE, create_transformer_with_hooks, create_text_sae
from .losses import SAELoss, compute_activation_statistics
from .utils import Logger, MetricTracker, CheckpointManager, EarlyStopping, setup_training_environment
import clip as openai_clip  # type: ignore

class SAETrainer:
    """Manage paired SAE training, validation, and checkpoint selection."""
    
    def __init__(self, config: Dict[str, Any]):
        """Build data loaders, models, and training utilities from a configuration."""
        self._validate_config(config)
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.logger, self.metric_tracker, self.checkpoint_manager = setup_training_environment(config)
        
        # Data setup supplies the tokenizer vocabulary needed to build the models.
        self._setup_data()
        self._setup_models()
        self._setup_training()
        
        if config.get('early_stopping', {}).get('enabled', False):
            early_stop_config = config['early_stopping']
            self.early_stopping = EarlyStopping(
                patience=early_stop_config.get('patience', 10),
                min_delta=early_stop_config.get('min_delta', 0.0),
                monitor_metric=early_stop_config.get('monitor_metric', 'val_total_loss'),
                mode=early_stop_config.get('mode', 'min')
            )
        else:
            self.early_stopping = None
        
        self.logger.info("SAE trainer initialized")

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        required = {
            "output_dir": config.get("output_dir"),
            "data.root_dir": config.get("data", {}).get("root_dir"),
            "data.train_ann_file": config.get("data", {}).get("train_ann_file"),
            "data.val_ann_file": config.get("data", {}).get("val_ann_file"),
            "model.text_backbone_weights": config.get("model", {}).get("text_backbone_weights"),
        }
        missing = [name for name, value in required.items() if not str(value or "").strip()]
        if missing:
            raise ValueError("Missing required training paths: " + ", ".join(missing))

        for name in (
            "root_dir",
            "train_ann_file",
            "val_ann_file",
        ):
            path = Path(config["data"][name])
            if not path.exists():
                raise FileNotFoundError(f"Configured data path not found: {path}")
        text_weights = Path(config["model"]["text_backbone_weights"])
        if not text_weights.is_file():
            raise FileNotFoundError(f"Text backbone checkpoint not found: {text_weights}")
    
    def _setup_data(self):
        """Build preprocessing and image-caption data loaders."""
        data_config = self.config['data']
        model_config = self.config['model']
        if model_config.get('use_clip_backbone', True):
            _, thisPreprocess = openai_clip.load("RN50", jit=False)
            self.image_transforms = thisPreprocess
            self.logger.info("Using CLIP image preprocessing")
        else: 
            self.image_transforms = ImageTransforms(
                resize_size=data_config.get('resize_size', 256),
                crop_size=data_config.get('crop_size', 224),
                imagenet_normalize=data_config.get('imagenet_normalize', True)
            )

        self.text_transforms = TextTransforms(
            tokenizer_name=data_config.get('tokenizer_name', 'bert-base-uncased'),
            max_length=data_config.get('max_length', 40),
            d_text=data_config.get('d_text', 768)
        )
        
        self.train_dataloader, self.val_dataloader = create_coco_dataloaders(
            root_dir=data_config['root_dir'],
            train_ann_file=data_config['train_ann_file'],
            val_ann_file=data_config['val_ann_file'],
            image_transforms=self.image_transforms,
            text_transforms=self.text_transforms,
            batch_size=data_config.get('batch_size', 64),
            num_workers=data_config.get('num_workers', 4),
            train_max_samples=data_config.get('train_max_samples'),
            val_max_samples=data_config.get('val_max_samples')
        )
        
        self.logger.info(f"Data loaders ready:")
        self.logger.info(f"  Training batches: {len(self.train_dataloader)}")
        self.logger.info(f"  Validation batches: {len(self.val_dataloader)}")
        self.logger.info(f"  Batch size: {data_config.get('batch_size', 64)}")
    
    def _setup_models(self):
        """Build the image and text backbones and their sparse autoencoders."""
        model_config = self.config['model']
        
        if model_config.get('use_clip_backbone', True):
            self.logger.info("Using a CLIP image backbone")
            self.image_backbone = create_clip_resnet_backbone({
                'clip_variant': model_config.get('clip_variant', 'RN50'),
                'pretrained': model_config.get('clip_pretrained', 'openai'),
                'hook_layer': model_config.get('hook_layer', 'layer4'),
                'freeze_backbone': model_config.get('freeze_backbone', True),
                'return_features': True,
                'finetuned_model_path': model_config.get('finetuned_model_path', None)
            }).to(self.device).eval()
        else:
            self.logger.info("Using a torchvision ResNet-50 image backbone")
            self.image_backbone = ResNetWithHooks(
                weights=model_config.get('resnet_weights', 'IMAGENET1K_V2'),
                hook_layer=model_config.get('hook_layer', 'layer4'),
                freeze_backbone=model_config.get('freeze_backbone', True)
            ).to(self.device).eval()
        
        self.image_sae = ImageSAE(
            input_dim=model_config.get('img_input_dim', 2048),
            hidden_dim=model_config.get('img_hidden_dim', 8192),
            k_sparse=model_config.get('img_k_sparse', 16),
            use_bias=model_config.get('use_bias', False),
            activation=model_config.get('activation', 'relu')
        ).to(self.device)
        
        vocab_size = self.text_transforms.get_vocab_size()
        self.text_backbone = create_transformer_with_hooks(
            vocab_size=vocab_size,
            config={
                'd_model': model_config.get('d_model', 512),
                'n_heads': model_config.get('n_heads', 8),
                'n_layers': model_config.get('n_layers', 6),
                'max_length': model_config.get('max_length', 40),
                'dropout': model_config.get('dropout', 0.1),
                'freeze_backbone': model_config.get('freeze_text_backbone', True),
                'return_features': True
            }
        ).to(self.device).eval()


        ckpt_path = model_config['text_backbone_weights']
        self.logger.info(f"Loading text backbone weights: {ckpt_path}")
        self.text_backbone.load_pretrained(ckpt_path)
        
        self.text_sae = create_text_sae(
            config={
                'd_model': model_config.get('d_model', 512),
                'sae_hidden_dim': model_config.get('txt_hidden_dim', 4096),
                'k_sparse': model_config.get('txt_k_sparse', 16),
                'use_bias': model_config.get('use_bias', False),
                'activation': model_config.get('activation', 'relu')
            }
        ).to(self.device)
        
        total_params = (sum(p.numel() for p in self.image_backbone.parameters()) +
                       sum(p.numel() for p in self.image_sae.parameters()) + 
                       sum(p.numel() for p in self.text_backbone.parameters()) +
                       sum(p.numel() for p in self.text_sae.parameters()))
        
        self.logger.log_model_info(self.image_sae, total_params)
        
        self.logger.info("Models ready")
    
    def _setup_training(self):
        """Configure losses, optimizer, scheduler, and training intervals."""
        train_config = self.config['training']
        
        loss_config = train_config.get('loss', {})
        self.criterion = SAELoss(
            lambda_rec=loss_config.get('lambda_rec', 1.0),
            lambda_sp=loss_config.get('lambda_sp', 0.1),
            lambda_kl=loss_config.get('lambda_kl', 0.01),
            reconstruction_type=loss_config.get('reconstruction_type', 'mse'),
            sparsity_type=loss_config.get('sparsity_type', 'l1'),
            cross_modal_type=loss_config.get('cross_modal_type', 'kl'),
            target_sparsity=loss_config.get('target_sparsity', 0.05),
            temperature=loss_config.get('temperature', 1.0),
            lambda_ue = loss_config.get('lambda_ue', 0.0),
            ue_even_weight = loss_config.get('ue_even_weight', 1.0)
        )
        
        optimizer_config = train_config.get('optimizer', {})
        
        # Optimize both SAEs and, when unfrozen, the text backbone.
        params_to_optimize = []
        params_to_optimize.extend(self.image_sae.parameters())
        params_to_optimize.extend(self.text_sae.parameters())
        
        if not self.text_backbone.freeze_backbone:
            params_to_optimize.extend(self.text_backbone.parameters())
        
        self.optimizer = optim.AdamW(
            params_to_optimize,
            lr=optimizer_config.get('lr', 3e-4),
            weight_decay=optimizer_config.get('weight_decay', 1e-2),
            betas=optimizer_config.get('betas', (0.9, 0.999))
        )
        
        scheduler_config = train_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=train_config.get('epochs', 100),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'onecycle':
            total_steps = len(self.train_dataloader) * train_config.get('epochs', 100)
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=optimizer_config.get('lr', 3e-4),
                total_steps=total_steps,
                pct_start=scheduler_config.get('pct_start', 0.3)
            )
        else:
            self.scheduler = None
        
        self.epochs = train_config.get('epochs', 100)
        self.grad_clip = train_config.get('grad_clip', 1.0)
        self.accumulation_steps = train_config.get('accumulation_steps', 1)
        self.log_interval = train_config.get('log_interval', 1000)
        self.val_interval = train_config.get('val_interval', 1)
        
        self.logger.info("Training components ready")
        self.logger.info(f"  Optimizer: {self.optimizer.__class__.__name__}")
        self.logger.info(f"  Scheduler: {self.scheduler.__class__.__name__ if self.scheduler else 'None'}")
        self.logger.info(f"  Epochs: {self.epochs}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train the SAEs for one epoch and return average batch metrics."""
        self.image_sae.train()
        self.text_sae.train()
        # Backbones remain in the evaluation mode set during model construction.
        
        epoch_losses = []
        epoch_components = {
            'total_loss': [],
            'reconstruction_loss': [],
            'cross_modal_loss': [],
            'img_rec_loss': [],
            'txt_rec_loss': [],
            'ue_loss': []
        }
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            img_features = self.image_backbone(images)
            spatial_features = img_features['spatial_features']  # [B, 49, 2048]

            img_outputs = self.image_sae(spatial_features)

            txt_features = self.text_backbone(input_ids, attention_mask)
            sequence_features = txt_features['sequence_features']  # [B, seq_len, d_model]

            txt_outputs = self.text_sae(sequence_features, attention_mask)

            loss, components = self.criterion(
                img_outputs, txt_outputs, return_components=True
            )

            # Scale each batch contribution for gradient accumulation.
            loss = loss / self.accumulation_steps
            
            if self.config.get('use_amp', False):
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.config.get('use_amp', False):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.image_sae.parameters()) + list(self.text_sae.parameters()),
                        self.grad_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.image_sae.parameters()) + list(self.text_sae.parameters()),
                        self.grad_clip
                    )
                    self.optimizer.step()
                
                # Advance OneCycleLR after optimizer updates.
                if self.scheduler and isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
            
            epoch_losses.append(loss.item() * self.accumulation_steps)
            for key, value in components.items():
                if key in epoch_components:
                    epoch_components[key].append(value)
            
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.accumulation_steps:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            if batch_idx % self.log_interval == 0:
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_dataloader)}, "
                    f"Loss: {loss.item() * self.accumulation_steps:.6f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
        
        # Advance other schedulers once per epoch.
        if self.scheduler and not isinstance(self.scheduler, OneCycleLR):
            self.scheduler.step()
        
        epoch_metrics = {
            'total_loss': sum(epoch_losses) / len(epoch_losses),
            'reconstruction_loss': sum(epoch_components['reconstruction_loss']) / len(epoch_components['reconstruction_loss']),
            'cross_modal_loss': sum(epoch_components['cross_modal_loss']) / len(epoch_components['cross_modal_loss']),
            'img_rec_loss': sum(epoch_components['img_rec_loss']) / len(epoch_components['img_rec_loss']),
            'txt_rec_loss': sum(epoch_components['txt_rec_loss']) / len(epoch_components['txt_rec_loss']),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Evaluate reconstruction losses and aggregated activation statistics."""
        self.image_sae.eval()
        self.text_sae.eval()
        
        val_losses = []
        val_components = {
            'total_loss': [],
            'reconstruction_loss': [],
            'img_rec_loss': [],
            'txt_rec_loss': [],
            'ue_loss': []
        }
        
        all_img_sparse = []
        all_txt_sparse = []
        
        progress_bar = tqdm(self.val_dataloader, desc=f"Validation {epoch}")
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            img_features = self.image_backbone(images)
            spatial_features = img_features['spatial_features']

            img_outputs = self.image_sae(spatial_features)

            txt_features = self.text_backbone(input_ids, attention_mask)
            sequence_features = txt_features['sequence_features']  # [B, seq_len, d_model]

            txt_outputs = self.text_sae(sequence_features, attention_mask)

            loss, components = self.criterion(
                img_outputs, txt_outputs, return_components=True
            )
            
            val_losses.append(loss.item())
            for key, value in components.items():
                if key in val_components:
                    val_components[key].append(value)
            
            # Collect sample-level activations for validation statistics.
            all_img_sparse.append(img_outputs['global_sparse'].cpu())
            all_txt_sparse.append(txt_outputs['global_sparse'].cpu())
            
            progress_bar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        val_metrics = {
            'total_loss': sum(val_losses) / len(val_losses),
            'reconstruction_loss': sum(val_components['reconstruction_loss']) / len(val_components['reconstruction_loss']),
            'img_rec_loss': sum(val_components['img_rec_loss']) / len(val_components['img_rec_loss']),
            'txt_rec_loss': sum(val_components['txt_rec_loss']) / len(val_components['txt_rec_loss']),
            'ue_loss': sum(val_components['ue_loss']) / len(val_components['ue_loss'])
        }
        
        img_sparse = torch.cat(all_img_sparse, dim=0)
        txt_sparse = torch.cat(all_txt_sparse, dim=0)
        activation_stats = compute_activation_statistics(img_sparse, txt_sparse)
        
        val_metrics.update(activation_stats)
        
        return val_metrics
    
    def train(self, resume_from: Optional[str] = None):
        """Run training and validation, optionally resuming a joint checkpoint."""
        start_epoch = 0
        
        if resume_from:
            checkpoint = self.checkpoint_manager.load_checkpoint(resume_from)
            
            self.image_sae.load_state_dict(checkpoint['model_state_dict']['image_sae'])
            self.text_backbone.load_state_dict(checkpoint['model_state_dict']['text_backbone'])
            self.text_sae.load_state_dict(checkpoint['model_state_dict']['text_sae'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            self.logger.info(f"Resuming training at epoch {start_epoch}")
        
        if self.config.get('use_amp', False):
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.logger.info("Starting training...")
        training_start_time = time.time()
        
        for epoch in range(start_epoch, self.epochs):
            epoch_start_time = time.time()
            
            train_metrics = self.train_epoch(epoch)
            
            if epoch % self.val_interval == 0:
                val_metrics = self.validate_epoch(epoch)
            else:
                val_metrics = {}
            
            epoch_time = time.time() - epoch_start_time
            
            self.metric_tracker.update(train_metrics, epoch, "train")
            if val_metrics:
                self.metric_tracker.update(val_metrics, epoch, "val")
            
            self.logger.log_epoch(epoch, train_metrics, "train", epoch_time)
            if val_metrics:
                self.logger.log_epoch(epoch, val_metrics, "val")
            
            # Select checkpoints using the configured validation metric.
            if val_metrics:
                monitor_score = val_metrics.get(
                    self.checkpoint_manager.monitor_metric.replace('val_', ''),
                    val_metrics.get('total_loss', float('inf'))
                )
                is_best = self.checkpoint_manager.should_save(monitor_score, epoch)
                
                model_state = {
                    'image_sae': self.image_sae.state_dict(),
                    'text_backbone': self.text_backbone.state_dict(),
                    'text_sae': self.text_sae.state_dict()
                }
                
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model_state=model_state,
                    model_config=self.config['model'],
                    optimizer_state=self.optimizer.state_dict(),
                    scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
                    metrics={**train_metrics, **val_metrics},
                    is_best=is_best
                )
            
            if self.early_stopping and val_metrics:
                if self.early_stopping(monitor_score, epoch):
                    break
            
            if epoch % 10 == 0:
                self.metric_tracker.save_metrics()
        
        total_time = time.time() - training_start_time
        self.logger.info(f"Training complete in {total_time:.2f}s")
        
        self.metric_tracker.save_metrics()
        self.metric_tracker.export_to_csv()
        
        if self.config.get('plot_metrics', True):
            self.metric_tracker.plot_metrics(
                ['total_loss', 'reconstruction_loss', 'sparsity_loss'],
                save_path=str(self.metric_tracker.save_dir / "training_curves.png")
            )
