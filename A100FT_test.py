project_name = "ISL_Finetuning"
sub_project_name = "RandomFT_NoFT_4layers"
run_name = "RandomFT_NoFT_4layers"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 1st one
#load_path = "/DATA3/vaibhav/isign/97CISLR/predictions_new/CISLR_Pretraining_PreTraining1_best_model_checkpoint_isignB4.pth"
#load_path = "/DATA3/vaibhav/isign/97CISLR/predictions_new/CISLR_Pretraining_PreTraining1_best_model_checkpoint.pth"
#load_path = "/DATA3/vaibhav/isign/97CISLR/predictions_new/CISLR_Pretraining_PreTraining2_best_model_checkpoint_isignB4.pth"
#load_path = "/DATA3/vaibhav/isign/97CISLR/predictions_new/CISLR_Pretraining_PreTraining2_best_model_checkpoint.pth"
load_path = ""

###################
# Change lr, beam width, layers, 
###################

# 0th one
#load_path = "/DATA7/vaibhav/tokenization/97CISLR/predictions_new/CISLR_Pretraining_Linear_schedulePT1_best_model_checkpoint_isignB4.pth"
#load_path = "/DATA7/vaibhav/tokenization/97CISLR/predictions_new/CISLR_Pretraining_Linear_schedulePT1_best_model_checkpoint.pth"
#load_path = "/DATA7/vaibhav/tokenization/97CISLR/predictions_new/CISLR_Pretraining_Linear_schedulePT2_best_model_checkpoint.pth"
#load_path = "/DATA7/vaibhav/tokenization/97CISLR/predictions_new/CISLR_Pretraining_Linear_schedulePT2_best_model_checkpoint_isignB4.pth"
#load_path = " "


import pandas as pd
import re
import numpy as np
import torch
import wandb
import random
import gc
import collections
import math

from tqdm import tqdm
from transformers import (
    BertConfig, BertModel,
    GPT2Config, GPT2LMHeadModel,
    EncoderDecoderModel,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    get_constant_schedule_with_warmup
)
from datasets import Dataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from helpers.dataloaders import FeatureVectorDataset_Isign
from helpers.bleu_cal import quick_bleu_metric
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set random seed for reproducibility
set_seed()


#Hyperparameters
learning_rate = 0.1 #6e-5         #Slower learning rate  
num_encoder_layers = 4
num_decoder_layers = 4
encoder_hidden_size = 512
decoder_hidden_size = 512
num_attention_heads = 8
dropout = 0.1
MAX_FRAMES = 300 # Max video frames.
max_position_embeddings_encoder = MAX_FRAMES
num_beams = 4 #3
warmup_steps_epocs = 10
batch_size = 64 #16
lr_scheduler_type = 'warmup_linear_constant_afterwards'
num_epochs = 1000
max_length_decoder = 128
vocab_size_decoder = 15000
num_keypoints = 152 # We have cherrypicked these
POSE_DIR = "/DATA7/vaibhav/isign/Data/iSign-poses_v1.1/"
tokenizer_file_path = "/DATA3/vaibhav/isign/97CISLR/tokenizer_file/target_tokenizer.json"
inf_steps = 2


hyperparameters = {
    'learning_rate': learning_rate,
    'num_encoder_layers': num_encoder_layers,
    'num_decoder_layers': num_decoder_layers,
    'encoder_hidden_size': encoder_hidden_size,
    'decoder_hidden_size': decoder_hidden_size,
    'num_attention_heads': num_attention_heads,
    'dropout': dropout,
    'max_position_embeddings_encoder': max_position_embeddings_encoder,
    'num_beams': num_beams,
    'warmup_steps_epocs': warmup_steps_epocs,
    'batch_size': batch_size,
    'lr_scheduler_type': lr_scheduler_type,
    'num_epochs': num_epochs,
    'max_length_decoder': max_length_decoder,
    'Max_frames_videos': MAX_FRAMES,
    'vocab_size_decoder': vocab_size_decoder,
    'num_keypoints': num_keypoints,
    'POSE_DIR': POSE_DIR,
    'max_frames': MAX_FRAMES,
    'inference_frame_steps': inf_steps,
    'sub_project_name': sub_project_name
}

wandb.init(project=project_name, name = run_name, config=hyperparameters)

# Data Loading
train_df = pd.read_csv('/DATA7/vaibhav/tokenization/train_split_unicode_filtered.csv')
eval_df = pd.read_csv('/DATA7/vaibhav/tokenization/val_split_unicode_filtered.csv')
test_df = pd.read_csv('/DATA7/vaibhav/tokenization/test_split_unicode_filtered.csv')
#train_df2 = pd.read_csv('/DATA7/vaibhav/tokenization/CISLR/train_split.csv')

# Tokenizer training
#all_sequences_target = train_df['text'].tolist() + train_df2['text'].tolist()

# tokenizer_model = models.BPE()
# tokenizer = Tokenizer(tokenizer_model)
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# trainer = trainers.BpeTrainer(
#     vocab_size=vocab_size_decoder,
#     special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
# )

# if not os.path.exists('tokenizer_file'):
#     os.makedirs('tokenizer_file')

# tokenizer.train_from_iterator(all_sequences_target, trainer=trainer)
# tokenizer.save("tokenizer_file/target_tokenizer.json")

tokenizer_target = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file_path)
tokenizer_target.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>"
})

# Prepare datasets
train_video_uids = train_df['uid'].tolist()
eval_video_uids = eval_df['uid'].tolist()
test_video_uids = test_df['uid'].tolist()

train_labels = [f'<s>{text}</s>' for text in train_df['text'].tolist()]
eval_labels = [f'<s>{text}</s>' for text in eval_df['text'].tolist()]
test_labels = [f'<s>{text}</s>' for text in test_df['text'].tolist()]

# Tokenize labels
train_labels = tokenizer_target(train_labels, max_length=max_length_decoder, padding="max_length", truncation=True)['input_ids']
eval_labels = tokenizer_target(eval_labels, max_length=max_length_decoder, padding="max_length", truncation=True)['input_ids']
test_labels = tokenizer_target(test_labels, max_length=max_length_decoder, padding="max_length", truncation=True)['input_ids']

# Create datasets
train_dataset = FeatureVectorDataset_Isign(train_video_uids, tokenizer_target, train_labels ,
                                           step_frames=inf_steps, add_noise = True)
test_dataset = FeatureVectorDataset_Isign(test_video_uids, tokenizer_target ,test_labels,
                                          step_frames=inf_steps, add_noise = True)
eval_dataset = FeatureVectorDataset_Isign(eval_video_uids, tokenizer_target, eval_labels,
                                          step_frames=inf_steps, add_noise = True)

# tp_tensor = torch.tensor([    0,   146,   124,  2562,   450,   144,  1785,   133,  8466,     2], dtype=torch.long)
# print(tokenizer_target.convert_ids_to_tokens(tp_tensor))
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

# Model initialization
encoder_config = BertConfig(
    hidden_size=encoder_hidden_size,
    num_hidden_layers=num_encoder_layers,
    num_attention_heads=num_attention_heads,
    hidden_dropout_prob=dropout,
    attention_probs_dropout_prob=dropout,
)
encoder = BertModel(encoder_config)
print(encoder_config)
decoder_config = GPT2Config(
    vocab_size=tokenizer_target.vocab_size,
    n_positions=max_length_decoder,
    n_embd=decoder_hidden_size,
    n_layer=num_decoder_layers,
    n_head=num_attention_heads,
    pad_token_id=tokenizer_target.pad_token_id,
    bos_token_id=tokenizer_target.bos_token_id,
    eos_token_id=tokenizer_target.eos_token_id,
    add_cross_attention=True,
    embd_pdrop=dropout,
    attn_pdrop=dropout,
    resid_pdrop=dropout
)
print(decoder_config)
decoder = GPT2LMHeadModel(decoder_config)

class FeatureProjection(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureProjection, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

feature_projection = FeatureProjection(num_keypoints, encoder_config.hidden_size)
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

model.config.decoder_start_token_id = tokenizer_target.bos_token_id
model.config.eos_token_id = tokenizer_target.eos_token_id
model.config.pad_token_id = tokenizer_target.pad_token_id
model.config.vocab_size = decoder_config.vocab_size
model.config.max_length = max_length_decoder

# Checkpoint paths
if not os.path.exists('predictions_new'):
    os.makedirs('predictions_new')

checkpoint_path = "predictions_new/"+project_name+'_'+sub_project_name+'_'+"checkpoint.pth"
best_checkpoint_path = "predictions_new/"+project_name+'_'+sub_project_name+'_'+"best_model_checkpoint.pth"

def save_checkpoint(model, feature_projection, optimizer, scheduler, epoch, best_val_B4, best_val_loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'feature_projection_state_dict': feature_projection.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_B4': best_val_B4,
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")
    print("*"*50)

def load_checkpoint(model, feature_projection, optimizer, scheduler, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        feature_projection.load_state_dict(checkpoint['feature_projection_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_B4 = checkpoint['best_val_B4']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))  # Backwards compatibility
        print(f"Checkpoint loaded, resuming from epoch {start_epoch}")
        print("*"*50)
        return start_epoch, best_val_B4, best_val_loss
    else:
        print("No checkpoint found, starting from scratch")
        return 0, 0.0, float('inf')

# Initialize model, optimizer, and scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
feature_projection.to(device)

optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(feature_projection.parameters()),
    weight_decay=1e-5,
    lr=learning_rate
)

# Calculate total steps for scheduler
total_steps = len(train_loader) * num_epochs
warmup_steps = len(train_loader) * warmup_steps_epocs

# Single scheduler with warmup

scheduler = get_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps
    #num_training_steps=total_steps  # Will maintain constant lr after warmup
)

wandb.watch(model, log="all", log_freq=100)

# Load checkpoint or pretrained weights
if os.path.exists(checkpoint_path):
    start_epoch, best_val_B4, best_val_loss = load_checkpoint(
        model, feature_projection, optimizer, scheduler, checkpoint_path
    )
    print("Loaded checkpoint model")
    print("*"*50)
elif os.path.exists(load_path):
    checkpoint = torch.load(load_path)
    #model.load_state_dict(torch.load(load_path, weights_only=True))
    model.load_state_dict(checkpoint['model_state_dict'])
    feature_projection.load_state_dict(checkpoint['feature_projection_state_dict'])
    start_epoch = 0
    best_val_B4 = 0.0
    best_val_loss = float('inf')
    print("Loaded pretrained model")
    print("*"*50)
else:
    print("No checkpoint or pretrained model found, starting from scratch")
    start_epoch = 0
    best_val_B4 = 0.0
    best_val_loss = float('inf')

# Training and Evaluation Loop
for epoch in range(start_epoch, num_epochs):
    # Training phase
    model.train()
    feature_projection.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        input_ids = feature_projection(input_ids)
        input_ids = input_ids.view(input_ids.size(0), -1, encoder_config.hidden_size)

        outputs = model(
            inputs_embeds=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(feature_projection.parameters()),
            max_norm=1.0
        )

        wandb.log({
            "learning_rate": scheduler.get_last_lr()[0],
            "step": epoch
        }, step=epoch)

        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        progress_bar.set_postfix({'Loss': loss.item()})
        
        # Evaluation phase (every 3000 steps)
        if (progress_bar.n + 1) % 3000 == 0:
            model.eval()
            feature_projection.eval()
            eval_loss = 0.0
            all_refs = []
            all_preds = []
            
            with torch.no_grad():
                eval_progress = tqdm(eval_loader, desc=f"Evaluating Epoch {epoch+1}")
                for eval_batch in eval_progress:
                    input_ids = eval_batch['input_ids'].to(device)
                    attention_mask = eval_batch['attention_mask'].to(device)
                    labels = eval_batch['labels'].to(device)
                    
                    input_ids = feature_projection(input_ids)
                    input_ids = input_ids.view(input_ids.size(0), -1, encoder_config.hidden_size)

                    outputs = model(
                        inputs_embeds=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    eval_loss += outputs.loss.item()
                    
                    # Generate predictions with improved parameters
                    generated_ids = model.generate(
                        inputs_embeds=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length_decoder,
                        num_beams=num_beams,
                        length_penalty=0.6,
                        no_repeat_ngram_size=3,
                        early_stopping=True
                    )
                    
                    # Process predictions and references
                    generated_ids = torch.where(
                        generated_ids == -100,
                        torch.tensor(tokenizer_target.pad_token_id).to(generated_ids.device),
                        generated_ids
                    )
                    labels = torch.where(
                        labels == -100,
                        torch.tensor(tokenizer_target.pad_token_id).to(labels.device),
                        labels
                    )
                    
                    preds = tokenizer_target.batch_decode(generated_ids, skip_special_tokens=True)
                    refs = tokenizer_target.batch_decode(labels, skip_special_tokens=True)
                    
                    ref_tokens = [ref.strip().split() for ref in refs]
                    pred_tokens = [pred.strip().split() for pred in preds]
                    
                    all_refs.extend([ref] for ref in ref_tokens)
                    all_preds.extend(pred_tokens)
            
            # Calculate metrics
            avg_eval_loss = eval_loss / len(eval_loader)
            bleu1, bleu2, bleu3, bleu4 = quick_bleu_metric(all_refs, all_preds, split='Validation')
            
            # Save best model
            if bleu4 > best_val_B4 or (bleu4 == best_val_B4 and avg_eval_loss < best_val_loss):
                best_val_B4 = bleu4
                best_val_loss = avg_eval_loss
                print('Saving best model checkpoint...')
                save_checkpoint(
                    model, feature_projection, optimizer, scheduler,
                    epoch, best_val_B4, best_val_loss, best_checkpoint_path
                )
                # Save predictions
                df = pd.DataFrame({
                    'Reference': [' '.join(ref[0]) for ref in all_refs],
                    'Prediction': [' '.join(pred) for pred in all_preds]
                })
                df.to_csv(f'predictions_new/{project_name}_{sub_project_name}_bestpredictions.csv', index=False)
            
            # Log metrics
            wandb.log({
                'epoch': epoch + 1,
                'val/eval_loss': avg_eval_loss,
                'val/bleu1': bleu1 * 100,
                'val/bleu2': bleu2 * 100,
                'val/bleu3': bleu3 * 100,
                'val/bleu4': bleu4 * 100
            })
            
            
            # Clean up memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # Resume training
            model.train()
            feature_projection.train()
    
    # End of epoch
    avg_train_loss = train_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs} - Average Training Loss: {avg_train_loss:.4f}')
    wandb.log({'epoch': epoch+1, 'train/train_loss': avg_train_loss})
    
    # Save regular checkpoint
    save_checkpoint(
        model, feature_projection, optimizer, scheduler,
        epoch, best_val_B4, best_val_loss, checkpoint_path
    )

# Final Testing Phase
print("Starting final testing phase...")
model.eval()
feature_projection.eval()
test_loss = 0.0
all_refs = []
all_preds = []

with torch.no_grad():
    test_progress = tqdm(test_loader, desc="Testing")
    for batch in test_progress:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        input_ids = feature_projection(input_ids)
        input_ids = input_ids.view(input_ids.size(0), -1, encoder_config.hidden_size)

        outputs = model(
            inputs_embeds=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        test_loss += outputs.loss.item()
        
        generated_ids = model.generate(
            inputs_embeds=input_ids,
            attention_mask=attention_mask,
            max_length=max_length_decoder,
            num_beams=num_beams,
            length_penalty=0.6,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        generated_ids = torch.where(
            generated_ids == -100,
            torch.tensor(tokenizer_target.pad_token_id).to(generated_ids.device),
            generated_ids
        )
        labels = torch.where(
            labels == -100,
            torch.tensor(tokenizer_target.pad_token_id).to(labels.device),
            labels
        )
        
        preds = tokenizer_target.batch_decode(generated_ids, skip_special_tokens=True)
        refs = tokenizer_target.batch_decode(labels, skip_special_tokens=True)
        
        ref_tokens = [ref.strip().split() for ref in refs]
        pred_tokens = [pred.strip().split() for pred in preds]
        
        all_refs.extend([ref] for ref in ref_tokens)
        all_preds.extend(pred_tokens)

# Calculate and log final test metrics
avg_test_loss = test_loss / len(test_loader)
bleu1, bleu2, bleu3, bleu4 = quick_bleu_metric(all_refs, all_preds, split='Test')

wandb.log({
    'test/test_loss': avg_test_loss,
    'test/bleu1_test': bleu1 * 100,
    'test/bleu2_test': bleu2 * 100,
    'test/bleu3_test': bleu3 * 100,
    'test/bleu4_test': bleu4 * 100
})

# Save final test predictions
df = pd.DataFrame({
    'Reference': [' '.join(ref[0]) for ref in all_refs],
    'Prediction': [' '.join(pred) for pred in all_preds]
})
df.to_csv(f'predictions_new/{project_name}_{sub_project_name}_final_test_predictions.csv', index=False)

# Finish the wandb run
wandb.finish()