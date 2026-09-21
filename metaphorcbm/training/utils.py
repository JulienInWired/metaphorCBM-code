"""Training logs, metrics, checkpoint selection, and early stopping."""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

from metaphorcbm.checkpoints import build_joint_sae_checkpoint, load_joint_sae_checkpoint


class Logger:
    """Write training messages to the console and a log file."""
    
    def __init__(self, 
                 log_dir: str,
                 experiment_name: str = "sae_experiment",
                 log_level: int = logging.INFO):
        """
        Configure logging for a named run.
        
        Args:
            log_dir: Directory for the log file.
            experiment_name: Run name used for the logger and filename.
            log_level: Python logging level.
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = self.log_dir / f"{experiment_name}.log"
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(experiment_name)
        self.start_time = time.time()
        
        self.logger.info(f"Run started: {experiment_name}")
        self.logger.info(f"Log directory: {self.log_dir}")
    
    def info(self, message: str):
        """Log an informational message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log a warning."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log an error."""
        self.logger.error(message)
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float], 
                  phase: str = "train", elapsed_time: Optional[float] = None):
        """Log epoch metrics and optional elapsed time in seconds."""
        metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        
        if elapsed_time is not None:
            self.logger.info(f"Epoch {epoch} [{phase}] - {metrics_str} - Time: {elapsed_time:.2f}s")
        else:
            self.logger.info(f"Epoch {epoch} [{phase}] - {metrics_str}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log configuration entries."""
        self.logger.info("Configuration:")
        for key, value in hparams.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_model_info(self, model, total_params: Optional[int] = None):
        """Log the model class and parameter counts."""
        if total_params is None:
            total_params = sum(p.numel() for p in model.parameters())
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model summary:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Model class: {model.__class__.__name__}")
    
    def get_elapsed_time(self) -> float:
        """Return seconds elapsed since logger initialization."""
        return time.time() - self.start_time


class MetricTracker:
    """Track metrics by phase and epoch, with JSON, CSV, and plot export."""
    
    def __init__(self, save_dir: Optional[str] = None):
        """Initialize metric storage and an optional output directory."""
        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(dict)
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def update(self, metrics: Dict[str, float], epoch: int, phase: str = "train"):
        """Append metric values and record them under the given epoch and phase."""
        for key, value in metrics.items():
            metric_key = f"{phase}_{key}"
            self.metrics[metric_key].append(value)
        
        self.epoch_metrics[epoch][phase] = metrics.copy()
    
    def get_metric(self, metric_name: str, phase: str = "train") -> List[float]:
        """Return all recorded values for a metric and phase."""
        metric_key = f"{phase}_{metric_name}"
        return self.metrics.get(metric_key, [])
    
    def get_latest(self, metric_name: str, phase: str = "train") -> Optional[float]:
        """Return the latest value, or None if no value has been recorded."""
        values = self.get_metric(metric_name, phase)
        return values[-1] if values else None
    
    def get_best(self, metric_name: str, phase: str = "train", 
                 mode: str = "min") -> Tuple[float, int]:
        """Return the best value and its index in the recorded metric sequence."""
        values = self.get_metric(metric_name, phase)
        if not values:
            return None, -1
        
        if mode == "min":
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        return values[best_idx], best_idx
    
    def compute_moving_average(self, metric_name: str, window: int = 10, 
                              phase: str = "train") -> List[float]:
        """Compute a trailing moving average, or return a sequence shorter than window."""
        values = self.get_metric(metric_name, phase)
        if len(values) < window:
            return values
        
        moving_avg = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            avg = np.mean(values[start_idx:i+1])
            moving_avg.append(avg)
        
        return moving_avg
    
    def plot_metrics(self, metric_names: List[str], 
                    phases: List[str] = ["train", "val"],
                    save_path: Optional[str] = None,
                    show_moving_avg: bool = False):
        """
        Plot recorded metrics for the selected phases.
        
        Args:
            metric_names: Metrics to plot.
            phases: Phase names to include.
            save_path: Optional image output path.
            show_moving_avg: Whether to overlay a moving average.
        """
        fig, axes = plt.subplots(len(metric_names), 1, 
                                figsize=(10, 4 * len(metric_names)))
        
        if len(metric_names) == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(metric_names):
            ax = axes[i]
            
            for phase in phases:
                values = self.get_metric(metric_name, phase)
                if values:
                    epochs = list(range(len(values)))
                    ax.plot(epochs, values, label=f"{phase}_{metric_name}", marker='o')
                    
                    if show_moving_avg and len(values) > 5:
                        moving_avg = self.compute_moving_average(metric_name, phase=phase)
                        ax.plot(epochs, moving_avg, 
                               label=f"{phase}_{metric_name}_ma", linestyle='--')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} history')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Metric plot saved to: {save_path}")
        
        plt.show()
    
    def save_metrics(self, filename: str = "metrics.json"):
        """Save metric sequences and epoch records as JSON."""
        if self.save_dir:
            save_path = self.save_dir / filename
        else:
            save_path = filename
        
        data = {
            'metrics': dict(self.metrics),
            'epoch_metrics': dict(self.epoch_metrics)
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Metrics saved to: {save_path}")
    
    def load_metrics(self, filename: str = "metrics.json"):
        """Restore metric sequences and epoch records from JSON."""
        if self.save_dir:
            load_path = self.save_dir / filename
        else:
            load_path = filename
        
        if os.path.exists(load_path):
            with open(load_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.metrics = defaultdict(list, data['metrics'])
            self.epoch_metrics = defaultdict(dict, data['epoch_metrics'])
            
            print(f"Metrics loaded from {load_path}")
        else:
            print(f"Metric file not found: {load_path}")
    
    def export_to_csv(self, filename: str = "metrics.csv"):
        """Export one CSV row per epoch with phase-prefixed metric columns."""
        if not self.epoch_metrics:
            print("No metrics to export")
            return
        
        rows = []
        for epoch, phases in self.epoch_metrics.items():
            row = {'epoch': epoch}
            for phase, metrics in phases.items():
                for metric, value in metrics.items():
                    row[f"{phase}_{metric}"] = value
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if self.save_dir:
            save_path = self.save_dir / filename
        else:
            save_path = filename
        
        df.to_csv(save_path, index=False)
        print(f"Metrics exported to: {save_path}")


class CheckpointManager:
    """Save the best joint SAE checkpoint according to a monitored metric."""
    
    def __init__(self, 
                 checkpoint_dir: str,
                 max_keep: int = 10,
                 monitor_metric: str = "val_total_loss",
                 mode: str = "min"):
        """
        Configure checkpoint output and selection.
        
        Args:
            checkpoint_dir: Directory for best_model.pth.
            max_keep: Unused; only the best checkpoint is stored.
            monitor_metric: Metric used to select the best model.
            mode: 'min' or 'max'.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.best_epoch = -1
        
        print(f"Checkpoint manager initialized:")
        print(f"  Output directory: {self.checkpoint_dir}")
        print(f"  Monitored metric: {monitor_metric} ({mode})")
        print(f"  Saving the best checkpoint to best_model.pth")
    
    def save_checkpoint(self, 
                       epoch: int,
                       model_state: Dict[str, Any],
                       model_config: Dict[str, Any],
                       optimizer_state: Dict[str, Any],
                       scheduler_state: Optional[Dict[str, Any]] = None,
                       metrics: Optional[Dict[str, float]] = None,
                       is_best: bool = False,
                       extra_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Write a joint checkpoint when is_best is True.
        
        Args:
            epoch: Epoch index.
            model_state: State dictionaries for the joint model components.
            model_config: Configuration needed to reconstruct the models.
            optimizer_state: Optimizer state dictionary.
            scheduler_state: Optional scheduler state dictionary.
            metrics: Optional metric values.
            is_best: Whether this checkpoint replaces the best model.
            extra_info: Optional checkpoint metadata.
            
        Returns:
            Saved checkpoint path, or an empty string when no file is written.
        """
        if not is_best:
            return ""
        
        checkpoint = build_joint_sae_checkpoint(
            epoch=epoch,
            model_state=model_state,
            model_config=model_config,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            metrics=metrics,
            extra_info=extra_info,
        )
        
        best_file = self.checkpoint_dir / "best_model.pth"
        
        torch.save(checkpoint, best_file)
        
        print(f"Best model updated (epoch {epoch}): {best_file}")
        
        return str(best_file)
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Load a joint checkpoint on CPU, defaulting to best_model.pth."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "best_model.pth"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = load_joint_sae_checkpoint(checkpoint_path, map_location='cpu')
        print(f"Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint
    
    def should_save(self, current_score: float, epoch: int) -> bool:
        """Update the best score and return whether the current model improves it."""
        is_better = False
        
        if self.mode == "min":
            is_better = current_score < self.best_score
        else:
            is_better = current_score > self.best_score
        
        if is_better:
            self.best_score = current_score
            self.best_epoch = epoch
            return True
        
        return False
    
    def list_checkpoints(self) -> List[str]:
        """Return the best checkpoint path if it exists."""
        best_model_file = self.checkpoint_dir / "best_model.pth"
        if best_model_file.exists():
            return [str(best_model_file)]
        return []
    
    def get_best_checkpoint_info(self) -> Dict[str, Any]:
        """Return the tracked best score, epoch, and selection settings."""
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'monitor_metric': self.monitor_metric,
            'mode': self.mode
        }


class EarlyStopping:
    """Stop after a configured number of checks without sufficient improvement."""
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.0,
                 monitor_metric: str = "val_total_loss",
                 mode: str = "min"):
        """
        Configure the stopping criterion.
        
        Args:
            patience: Number of consecutive checks without improvement.
            min_delta: Minimum score change counted as an improvement.
            monitor_metric: Name used to report the monitored score.
            mode: 'min' or 'max'.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
        
        print(f"Early stopping initialized:")
        print(f"  Patience: {patience}")
        print(f"  Monitored metric: {monitor_metric} ({mode})")
        print(f"  Minimum improvement: {min_delta}")
    
    def __call__(self, current_score: float, epoch: int) -> bool:
        """Update the monitored score and return whether training should stop."""
        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            print(f"Early stopping at epoch {epoch}")
            print(f"Best {self.monitor_metric}: {self.best_score:.6f}")
            return True
        
        return False


def setup_training_environment(config: Dict[str, Any]) -> Tuple[Logger, MetricTracker, CheckpointManager]:
    """Create run directories and return a logger, metric tracker, and checkpoint manager."""
    experiment_name = config.get('experiment_name', 'sae_experiment')
    output_value = config.get('output_dir')
    if not str(output_value or '').strip():
        raise ValueError("output_dir must be provided in the training configuration")
    output_dir = Path(output_value)
    
    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logger = Logger(
        log_dir=str(exp_dir / "logs"),
        experiment_name=experiment_name
    )
    
    metric_tracker = MetricTracker(
        save_dir=str(exp_dir / "metrics")
    )
    
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(exp_dir / "checkpoints"),
        max_keep=config.get('max_keep_checkpoints', 5),
        monitor_metric=config.get('monitor_metric', 'val_total_loss'),
        mode=config.get('monitor_mode', 'min')
    )
    
    config_file = exp_dir / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Training environment ready")
    logger.info(f"Run directory: {exp_dir}")
    logger.log_hyperparameters(config)
    
    return logger, metric_tracker, checkpoint_manager 
