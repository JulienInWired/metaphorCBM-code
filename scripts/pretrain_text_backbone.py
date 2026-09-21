"""Pretrain a transformer text backbone with masked language modeling on captions."""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import random
import argparse
from typing import Dict, List, Tuple, Optional
import math

from metaphorcbm.data.transforms import TextTransforms
from metaphorcbm.models.sae_text import TransformerEncoder


class MLMDataset(Dataset):
    """Caption dataset with dynamically sampled masked-language targets."""
    
    def __init__(self, 
                 ann_file: str,
                 text_transforms: TextTransforms,
                 mask_prob: float = 0.15,
                 max_samples: Optional[int] = None,
                 seed: int = 42):
        """
        Load captions and configure token masking.
        
        Args:
            ann_file: COCO-format caption annotations.
            text_transforms: Tokenizer and text preprocessing callable.
            mask_prob: Fraction of eligible tokens selected as prediction targets.
            max_samples: Optional caption count limit.
            seed: Python and PyTorch random seed.
        """
        self.text_transforms = text_transforms
        self.mask_prob = mask_prob
        self.mask_token_id = text_transforms.tokenizer.mask_token_id
        self.pad_token_id = text_transforms.tokenizer.pad_token_id
        
        random.seed(seed)
        torch.manual_seed(seed)
        
        print(f"Loading annotations: {ann_file}")
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.captions = []
        for ann in data['annotations']:
            caption = ann['caption'].strip()
            if len(caption) > 0:
                self.captions.append(caption)
        
        if max_samples is not None and max_samples < len(self.captions):
            self.captions = random.sample(self.captions, max_samples)
        
        print(f"Loaded {len(self.captions)} captions")
    
    def __len__(self) -> int:
        return len(self.captions)
    
    def mask_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample prediction targets using BERT-style token masking.
        
        Args:
            input_ids: Token IDs [seq_len].
            attention_mask: Valid-token mask [seq_len].
            
        Returns:
            Masked token IDs and target labels, with -100 at ignored positions.
        """
        input_ids = input_ids.clone()
        labels = input_ids.clone()
        
        # Exclude boundary and padding tokens from prediction targets.
        special_tokens = {
            self.text_transforms.tokenizer.cls_token_id,
            self.text_transforms.tokenizer.sep_token_id,
            self.text_transforms.tokenizer.pad_token_id
        }
        
        maskable_positions = []
        for i, (token_id, mask) in enumerate(zip(input_ids, attention_mask)):
            if mask == 1 and token_id.item() not in special_tokens:
                maskable_positions.append(i)
        
        num_to_mask = max(1, int(len(maskable_positions) * self.mask_prob))
        mask_positions = random.sample(maskable_positions, min(num_to_mask, len(maskable_positions)))
        
        # Replace targets with [MASK] (80%), a random token (10%), or no change (10%).
        for pos in mask_positions:
            rand = random.random()
            if rand < 0.8:
                input_ids[pos] = self.mask_token_id
            elif rand < 0.9:
                input_ids[pos] = random.randint(0, len(self.text_transforms.tokenizer) - 1)
        
        # Ignore positions that were not selected as prediction targets.
        for i in range(len(labels)):
            if i not in mask_positions:
                labels[i] = -100
        
        return input_ids, labels
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Tokenize one caption and sample its masked-language targets."""
        caption = self.captions[idx]
        
        text_data = self.text_transforms(caption)
        input_ids = text_data['input_ids']
        attention_mask = text_data['attention_mask']
        
        masked_input_ids, labels = self.mask_tokens(input_ids, attention_mask)
        
        return {
            'input_ids': masked_input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class MLMWrapper(nn.Module):
    """Transformer encoder with a vocabulary projection tied to token embeddings."""
    
    def __init__(self, vocab_size: int, **encoder_kwargs):
        """Build the encoder and language-model head for the given vocabulary."""
        super().__init__()
        
        self.encoder = TransformerEncoder(vocab_size=vocab_size, **encoder_kwargs)
        self.lm_head = nn.Linear(encoder_kwargs.get('d_model', 512), vocab_size, bias=False)
        
        # Tie the output projection to the input token embeddings.
        self.lm_head.weight = self.encoder.token_embedding.weight
        
        print(f"MLMWrapper initialized:")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Model dimension: {encoder_kwargs.get('d_model', 512)}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return vocabulary logits for every token position."""
        hidden_states = self.encoder(input_ids, attention_mask)  # [B, seq_len, d_model]
        
        logits = self.lm_head(hidden_states)  # [B, seq_len, vocab_size]
        
        return logits


def compute_mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute token cross-entropy, ignoring target positions labeled -100."""
    shift_logits = logits.view(-1, logits.size(-1))  # [B*seq_len, vocab_size]
    shift_labels = labels.view(-1)  # [B*seq_len]
    
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    
    return loss


def train_epoch(model: MLMWrapper, 
                dataloader: DataLoader, 
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int) -> float:
    """Train for one epoch and return the mean batch loss."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, attention_mask)
        
        loss = compute_mlm_loss(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'ppl': f'{math.exp(min(avg_loss, 10)):.2f}'
        })
    
    return total_loss / num_batches


def evaluate(model: MLMWrapper, 
             dataloader: DataLoader, 
             device: torch.device) -> float:
    """Return the mean masked-language loss over evaluation batches."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = compute_mlm_loss(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def build_epoch_scheduler(
    optimizer: optim.Optimizer,
    *,
    train_batches: int,
    epochs: int,
    warmup_steps: int,
) -> optim.lr_scheduler.LambdaLR:
    """Build an epoch-stepped warmup/cosine schedule with a batch-count horizon."""
    schedule_horizon = train_batches * epochs

    def lr_lambda(epoch_index: int) -> float:
        if epoch_index < warmup_steps:
            return epoch_index / max(warmup_steps, 1)
        denominator = max(schedule_horizon - warmup_steps, 1)
        progress = min((epoch_index - warmup_steps) / denominator, 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description='Pretrain a transformer text backbone with masked language modeling')
    parser.add_argument('--data_root', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--train_ann', type=str, default='annotations/captions_train2017.json', help='Training caption annotations, relative to data_root')
    parser.add_argument('--val_ann', type=str, default='annotations/captions_val2017.json', help='Validation caption annotations, relative to data_root')
    parser.add_argument('--output_dir', type=str, required=True, help='Output checkpoint directory')
    parser.add_argument('--max_samples', type=int, default=None, help='Training caption limit; validation uses one tenth of this count')
    
    # Model dimensions.
    parser.add_argument('--d_model', type=int, default=512, help='Token feature dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--max_length', type=int, default=40, help='Maximum token sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    
    # Training settings.
    parser.add_argument('--batch_size', type=int, default=256, help='Captions per batch')
    parser.add_argument('--lr', type=float, default=2e-3, help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='AdamW weight decay')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=3000, help='Warmup length in scheduler updates, one update per epoch')
    parser.add_argument('--mask_prob', type=float, default=0.15, help='Fraction of eligible tokens selected for prediction')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data-loading workers')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    text_transforms = TextTransforms(
        tokenizer_name="bert-base-uncased",
        max_length=args.max_length,
        add_special_tokens=True
    )
    vocab_size = text_transforms.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    train_dataset = MLMDataset(
        ann_file=os.path.join(args.data_root, args.train_ann),
        text_transforms=text_transforms,
        mask_prob=args.mask_prob,
        max_samples=args.max_samples
    )
    
    val_dataset = MLMDataset(
        ann_file=os.path.join(args.data_root, args.val_ann),
        text_transforms=text_transforms,
        mask_prob=args.mask_prob,
        max_samples=args.max_samples // 10 if args.max_samples else None
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    model = MLMWrapper(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        max_length=args.max_length,
        dropout=args.dropout
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98)
    )
    
    scheduler = build_epoch_scheduler(
        optimizer,
        train_batches=len(train_dataloader),
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
    )
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_dataloader, optimizer, device, epoch)
        
        val_loss = evaluate(model, val_dataloader, device)
        
        # Advance the schedule once after each epoch.
        scheduler.step()
        
        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Training loss: {train_loss:.4f} (perplexity: {math.exp(min(train_loss, 10)):.2f})")
        print(f"  Validation loss: {val_loss:.4f} (perplexity: {math.exp(min(val_loss, 10)):.2f})")
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save encoder weights without the tied language-model head.
            torch.save(
                model.encoder.state_dict(),
                os.path.join(args.output_dir, 'text_backbone_pretrained.pt')
            )
            print(f"  Best model saved (validation loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping: no validation improvement for {patience} epochs")
                break
        
        print("-" * 50)
    
    print("Pretraining complete")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {os.path.join(args.output_dir, 'text_backbone_pretrained.pt')}")


if __name__ == '__main__':
    main()
