# Get the full dataset, save tokenizer.
# Uncomment best_checkpoint_path_isignB4 in the load path if you want to load from checkpoint

project_name = "CISLR_Pretraining"
sub_project_name = "DO_MT_GN_RF_MixIsign_PT1"
run_name = "DO_MT_GN_RF_MixIsign_PT1"

# Gausian Noise , Random frame sampling , Isign mixed with CISLR linearly

randomize_word_order = False
steps_for_100percentIsign = 60000
import os
# # Set the visible GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import pandas as pd
import re
import numpy as np
import torch
import wandb
import random
import gc
import collections
import math
import ast
import collections
import math
import time
import torch.nn as nn

from tqdm import tqdm
from transformers import (
    BertConfig, BertModel,
    GPT2Config, GPT2LMHeadModel,GPT2Model,
    EncoderDecoderModel,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    get_constant_schedule_with_warmup
)
from datasets import Dataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from helpers.bleu_cal import quick_bleu_metric
from helpers.dataloaders import DecoderOnlyDatasetCISLR, DecoderOnlyDatasetIsign
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from itertools import cycle


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set random seed for reproducibility
set_seed()

def get_threshold(current_step, total_steps):
    return min(current_step / total_steps, 0.9)

#Hyperparameters here now 
learning_rate = 3e-4 #3e-4 #Slower learning rate for finetuning
hidden_size = 512
num_heads = 8
num_layers = 4
num_beams = 3
# dropout = 0.1
MAX_FRAMES = 300  # Max video frames.
max_position_embeddings_encoder = MAX_FRAMES
warmup_steps_ratio = 0.1
batch_size = 16 #64 #256
#gradient_accumulation_steps = 1 not used yet
lr_scheduler_type = 'warmup_linear_constant_afterwards'
num_epochs = 1
max_length_decoder = 128
vocab_size_decoder = 15000
num_keypoints = 152 # We have cherrypicked these
steps_for_100percentIsign = 60000
POSE_DIR = "/DATA7/vaibhav/tokenization/CISLR/CISLR_v1.5-a_videos_poses/"
POSE_DIR_ISIGN = "/DATA7/vaibhav/isign/Data/iSign-poses_v1.1/"


hyperparameters = { 'learning_rate': learning_rate, 
                    'hidden_size': hidden_size,
                    'num_heads': num_heads,
                    'num_layers': num_layers,
                    'num_beams': num_beams,
                    'steps_for_100percentIsign': steps_for_100percentIsign,
                    'max_position_embeddings_encoder': max_position_embeddings_encoder,
                    'warmup_steps_ratio': warmup_steps_ratio,
                    'batch_size': batch_size,
                    'lr_scheduler_type': lr_scheduler_type,
                    'num_epochs': num_epochs,
                    'max_length_decoder': max_length_decoder,
                    'Max_frames_videos': MAX_FRAMES,
                    'vocab_size_decoder': vocab_size_decoder,
                    'num_keypoints': num_keypoints,
                    'POSE_DIR': POSE_DIR,
                    'POSE_DIR_ISIGN': POSE_DIR_ISIGN,
                    'randomize_word_order': randomize_word_order
                    ,'sub_project_name': sub_project_name}




wandb.init(project=project_name, name=run_name, config = hyperparameters)

#wandb.init(project=project_name, config = hyperparameters, id="d9vu0xp8", resume="must")

#wandb.init(project=project_name, config = hyperparameters, id="7ike4lk8", resume="must")


train_df = pd.read_csv('/DATA3/vaibhav/isign/PretrainingISL/train_MT.csv')
eval_df = pd.read_csv('/DATA3/vaibhav/isign/PretrainingISL/val_MT.csv')
test_df = pd.read_csv('/DATA3/vaibhav/isign/PretrainingISL/test_MT.csv')

# train_df = train_df.sample(n=1000)
# eval_df = eval_df.sample(n=1000)

eval_df2 = pd.read_csv('/DATA7/vaibhav/tokenization/val_split_unicode_filtered.csv')
train_df2 = pd.read_csv('/DATA3/vaibhav/isign/PretrainingISL/isign_new.csv')

# train_df2 = train_df2.sample(n=1000)
# eval_df2 = eval_df2.sample(n=1000)

# Step 2: Train Tokenizers
# Combine source and target sequences for a joint tokenizer
#all_sequences = train_df['SENTENCE_UNICIDE'].tolist() + train_df['text'].tolist()
    
all_sequences_target = train_df['text'].tolist() + train_df2['text'].tolist()


# Initialize and train the tokenizer
tokenizer_model = models.BPE()
tokenizer = Tokenizer(tokenizer_model)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size_decoder,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

tokenizer.train_from_iterator(all_sequences_target, trainer=trainer)
# Save the tokenizer
#Make tokenizer_file if it does not exist

if not os.path.exists('tokenizer_file'):
    os.makedirs('tokenizer_file')


######################### Save tokenizer.
tokenizer.save("tokenizer_file/target_tokenizer_decoderonly.json")

# Load the tokenizer as a PreTrainedTokenizerFast
tokenizer_target = PreTrainedTokenizerFast(tokenizer_file="tokenizer_file/target_tokenizer_decoderonly.json")
tokenizer_target.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>"
})
tokenizer_target.add_special_tokens({'additional_special_tokens': ['<pose>', '<English>']})




# Extract video UIDs and labels

print('Extracting video UIDs and labels...')
train_video_uids = train_df['uid_list'].apply(ast.literal_eval).tolist()
eval_video_uids = eval_df['uid_list'].apply(ast.literal_eval).tolist()
test_video_uids = test_df['uid_list'].apply(ast.literal_eval).tolist()

train2_video_uids = train_df2['uid'].tolist()
eval2_video_uids = eval_df2['uid'].tolist()

# print('Appending <s> and </s> to labels...')
train_labels = [f'<s>{text}</s>' for text in train_df['text'].tolist()]
eval_labels = [f'<s>{text}</s>' for text in eval_df['text'].tolist()]
test_labels = [f'<s>{text}</s>' for text in test_df['text'].tolist()]

train2_labels = [f'<s>{text}</s>' for text in train_df2['text'].tolist()]
eval2_labels = [f'<s>{text}</s>' for text in eval_df2['text'].tolist()]


def tokenize_in_batches(texts, tokenizer, max_length, batch_size=1000):
    all_tokens = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(
            batch, 
            max_length=max_length, 
            padding="max_length", 
            truncation=True
        )['input_ids']
        all_tokens.extend(tokens)
    return all_tokens


# Tokenize labels
print('Tokenizing labels...')
train_labels = tokenize_in_batches(train_labels, tokenizer_target, max_length_decoder)
#train_labels = tokenizer_target(train_labels, max_length=max_length_decoder, padding="max_length", truncation=True)['input_ids']
eval_labels = tokenizer_target(eval_labels, max_length=max_length_decoder, padding="max_length", truncation=True)['input_ids']
test_labels = tokenizer_target(test_labels, max_length=max_length_decoder, padding="max_length", truncation=True)['input_ids']

train2_labels = tokenizer_target(train2_labels, max_length=max_length_decoder, padding="max_length", truncation=True)['input_ids']
eval2_labels = tokenizer_target(eval2_labels, max_length=max_length_decoder, padding="max_length", truncation=True)['input_ids']

# Create datasets
print('Creating datasets...')
train_dataset = DecoderOnlyDatasetCISLR(train_video_uids,tokenizer_target,
                                      randomize_word_order, MAX_FRAMES, POSE_DIR,
                                      train_labels, step_frames=None, add_noise = True)
test_dataset = DecoderOnlyDatasetCISLR(test_video_uids,tokenizer_target, 
                                    randomize_word_order, MAX_FRAMES, POSE_DIR,
                                    test_labels,step_frames=None, add_noise = True)
eval_dataset = DecoderOnlyDatasetCISLR(eval_video_uids,tokenizer_target, 
                                    randomize_word_order, MAX_FRAMES, POSE_DIR,
                                    eval_labels,step_frames=None, add_noise = True)

train2_dataset = DecoderOnlyDatasetIsign(train2_video_uids, tokenizer_target,MAX_FRAMES, POSE_DIR_ISIGN,
                                         train2_labels , step_frames=None, add_noise = False)

eval2_dataset = DecoderOnlyDatasetIsign(eval2_video_uids, tokenizer_target, MAX_FRAMES, POSE_DIR_ISIGN,
                                        eval2_labels, step_frames=None, add_noise = False)

# tp_tensor = torch.tensor([    0,   146,   124,  2562,   450,   144,  1785,   133,  8466,     2], dtype=torch.long)
# print(tokenizer_target.convert_ids_to_tokens(tp_tensor))
# Create DataLoaders
print('Creating DataLoaders...')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=2)


isign_loader = DataLoader(train2_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
eval2_loader = DataLoader(eval2_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=2)


isign_loader_cycle = cycle(isign_loader)  # To cycle through ISIGN when exhausted
cislr_loader_cycle = cycle(train_loader)  # To cycle through CISLR when exhausted

class SignLanguageGPT(nn.Module):
    def __init__(self, gpt_model, pose_dim, hidden_size, max_pose_len=MAX_FRAMES):
        super().__init__()
        self.gpt = gpt_model
        self.pose_projection = nn.Linear(pose_dim, hidden_size)
        self.max_pose_len = max_pose_len
        self.hidden_size = hidden_size

    def forward(self, input_ids, pose_features, attention_mask, pose_mask, labels=None):
        batch_size = pose_features.size(0)

        # Split input_ids into components
        pose_token = input_ids[:, 0:1]  # <pose>
        english_token_idx = self.max_pose_len + 1  # After pose token and pad tokens
        english_token = input_ids[:, english_token_idx:english_token_idx+1]  # <English>
        text_portion = input_ids[:, english_token_idx+1:]  # Rest is text
    
        # Get embeddings separately
        pose_token_embeds = self.gpt.transformer.wte(pose_token)
        english_token_embeds = self.gpt.transformer.wte(english_token)
        text_embeds = self.gpt.transformer.wte(text_portion)  # Limit text to 128
        
        # Project pose features
        projected_pose = self.pose_projection(pose_features)  # [batch_size, max_pose_len, hidden_size]
        
        # print(f"Shapes - pose_token: {pose_token_embeds.shape}, pose: {projected_pose.shape}, "
        #   f"english: {english_token_embeds.shape}, text: {text_embeds.shape}")
    

        combined_embeds = torch.cat([
        pose_token_embeds,      # <pose>
        projected_pose,         # pose features
        english_token_embeds,   # <English>
        text_embeds            # text (truncated)
    ], dim=1)

        # Verify sequence length
        assert combined_embeds.size(1) <= self.gpt.config.n_positions, \
            f"Sequence length {combined_embeds.size(1)} exceeds maximum position embedding size {self.gpt.config.n_positions}"


        # Forward pass through GPT
        outputs = self.gpt(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Calculate maximum sequence length
max_sequence_length = (
    1 +                # <pose> token
    MAX_FRAMES +       # pose features
    1 +                # <English> token
    128               # max text length
)

# Initialize model
gpt_config = GPT2Config(
    vocab_size=len(tokenizer_target),
    n_positions=max_sequence_length,  # Set to total sequence length
    n_layer=num_layers,
    n_head=num_heads,
    n_embd=hidden_size
)

gpt_model = GPT2LMHeadModel(gpt_config).to(device)
model = SignLanguageGPT(gpt_model, pose_dim=num_keypoints, hidden_size=hidden_size).to(device)

def generate_from_pose(pose_features, model, tokenizer, max_length=128):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        # Project pose features
        if len(pose_features.shape) == 2:
            pose_features = pose_features.unsqueeze(0)
        pose_features = torch.tensor(pose_features, device=device)
        
        batch_size = pose_features.size(0)
        

        # Create initial input sequence
        input_ids = torch.full((batch_size, 1), tokenizer.convert_tokens_to_ids('<pose>'), device=device)
        pad_tokens = torch.full((batch_size, 300), tokenizer.pad_token_id, device=device)
        english_token = torch.full((batch_size, 1), tokenizer.convert_tokens_to_ids('<English>'), device=device)
        
        # Initialize with special tokens
        curr_input_ids = torch.cat([input_ids, pad_tokens, english_token], dim=1)
        curr_attention_mask = torch.ones_like(curr_input_ids)
        
        
        # pose_input = torch.tensor(pose).unsqueeze(0).to(device)
        # projected_pose = model.pose_projection(pose_input)
        
        
         # Generate sequence
        for _ in range(max_length):
            outputs = model(
                input_ids=curr_input_ids,
                pose_features=pose_features,
                attention_mask=curr_attention_mask,
                pose_mask=torch.ones(batch_size, 300, device=device)
            )
            
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            curr_input_ids = torch.cat([curr_input_ids, next_token.unsqueeze(1)], dim=1)
            curr_attention_mask = torch.cat([curr_attention_mask, torch.ones_like(next_token.unsqueeze(1))], dim=1)
            
            if (next_token == tokenizer.eos_token_id).all():
                break
        
        # Decode only text portion
        generated = curr_input_ids[:, 302:].cpu().tolist()
        return [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated]


# Load the best model checkpoint if it exists
# Create directory if it does not exist
if not os.path.exists('load_pretrained'):
    os.makedirs('load_pretrained')

if not os.path.exists('predictions_new'):
    os.makedirs('predictions_new')

load_path = ""
checkpoint_path = "predictions_new/"+project_name+'_'+sub_project_name+'_'+"checkpoint.pth"
best_checkpoint_path = "predictions_new/"+project_name+'_'+sub_project_name+'_'+"best_model_checkpoint.pth"
best_checkpoint_path_isignB1 = "predictions_new/"+project_name+'_'+sub_project_name+'_'+"best_model_checkpoint_isign.pth"
best_checkpoint_path_isignB4 = "predictions_new/"+project_name+'_'+sub_project_name+'_'+"best_model_checkpoint_isignB4.pth"
def save_checkpoint(model, optimizer, scheduler, epoch, best_val_B4, best_val_loss, checkpoint_path, current_step,
                    best_val_B4_isign, best_val_loss_isign, best_val_B1_isign, epoch_steps):
    checkpoint = {
        'epoch': epoch,
        'current_step' : current_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_B4': best_val_B4,
        'best_val_loss': best_val_loss,
        'best_val_B4_isign': best_val_B4_isign,
        'best_val_loss_isign': best_val_loss_isign,
        'best_val_B1_isign': best_val_B1_isign,
        'epoch_steps': epoch_steps
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {current_step}")
    print("*"*50)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        current_step = checkpoint['current_step']
        best_val_B4 = checkpoint['best_val_B4']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))  # Backwards compatibility
        best_val_B4_isign = checkpoint['best_val_B4_isign']
        best_val_B1_isign = checkpoint['best_val_B1_isign']
        best_val_loss_isign = checkpoint.get('best_val_loss_isign', float('inf'))  # Backwards compatibility
        epoch_steps = checkpoint['epoch_steps']
        print(f"Checkpoint loaded, resuming from epoch {start_epoch}")
        print("*"*50)
        return start_epoch, best_val_B4, best_val_loss, best_val_B4_isign, best_val_loss_isign, best_val_B1_isign, epoch_steps
    else:
        print("No checkpoint found, starting from scratch")
        return 0, 0.0, float('inf')


# Initialize model, optimizer, and scheduler
#model.to(device)
optimizer = torch.optim.AdamW(
    list(model.parameters()) ,
    weight_decay=1e-5,
    lr=learning_rate
)

# Calculate total steps for scheduler
total_steps = len(train_loader)  
# Set warmup to 10% of total steps
warmup_steps = int(warmup_steps_ratio * total_steps)


# Create scheduler with linear warmup and constant afterwards

scheduler = get_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    #num_training_steps=total_steps  # Will maintain constant lr after warmup
)

#wandb.watch(model, log="all", log_freq=100)

epoch_steps = 0
# Load checkpoint or pretrained weights
if os.path.exists(""): #best_checkpoint_path_isignB4
    start_epoch, best_val_B4, best_val_loss, best_val_B4_isign, best_val_loss_isign, best_val_B1_isign, epoch_steps = load_checkpoint(
        model, optimizer, scheduler, best_checkpoint_path_isignB4
    )
    start_epoch = 0
    print("Loaded best model checkpoint IsignB4")
    print("*"*50)

elif os.path.exists(load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Reset epoch and metrics but keep scheduler state
    start_epoch = 0
    best_val_B4 = 0.0
    best_val_B4_isign = 0.0
    best_val_B1_isign = 0.0
    best_val_loss_isign = float('inf')
    best_val_loss = float('inf')
    print("Loaded pretrained model")
    print("*"*50)
else:
    print("No checkpoint or pretrained weights found, starting from scratch")
    print("*"*50)
    start_epoch = 0
    best_val_B4 = 0.0
    best_val_B4_isign = 0.0
    best_val_B1_isign = 0.0
    best_val_loss_isign = float('inf')
    best_val_loss = float('inf')

def model_eval(eval_loader, log_what, best_val_B4,best_val_loss,best_val_B4_isign,
               best_val_B1_isign,best_val_loss_isign,counter, current_step,epoch_steps, save_model=False):
    model.eval()
    eval_loss = 0.0
    all_refs = []
    all_preds = []
    
    with torch.no_grad():
        eval_progress = tqdm(eval_loader, desc=f"Evaluating Epoch {epoch+1}")
        for eval_batch in eval_progress:
    
            pose_features = eval_batch['pose_features'].to(device)
            labels = eval_batch['labels'].to(device)
            
            input_ids = eval_batch['input_ids'].to(device)
            pose_features = eval_batch['pose_features'].to(device) 
            attention_mask = eval_batch['attention_mask'].to(device)
            pose_mask = eval_batch['pose_mask'].to(device)
            
            # Split input_ids into components
            pose_token = input_ids[:, 0:1]  # <pose>
            english_token_idx = MAX_FRAMES + 1  # After pose token and pad tokens
            english_token = input_ids[:, english_token_idx:english_token_idx+1]  # <English>
            text_portion = input_ids[:, english_token_idx+1:]  # Rest is text
        
            # Get embeddings separately
            pose_token_embeds = model.gpt.transformer.wte(pose_token)
            english_token_embeds = model.gpt.transformer.wte(english_token)
            text_embeds = model.gpt.transformer.wte(text_portion)  # Limit text to 128
            
            # Project pose features
            projected_pose = model.pose_projection(pose_features)  # [batch_size, max_pose_len, hidden_size]
            
            # print(f"Shapes - pose_token: {pose_token_embeds.shape}, pose: {projected_pose.shape}, "
            #   f"english: {english_token_embeds.shape}, text: {text_embeds.shape}")
        

            combined_embeds = torch.cat([
            pose_token_embeds,      # <pose>
            projected_pose,         # pose features
            english_token_embeds,   # <English>
            text_embeds            # text (truncated)
        ], dim=1)


            # Generate predictions using the generate_from_pose function
            #preds = generate_from_pose(pose_features.cpu().numpy(), model, tokenizer_target, max_length=128)
            generated_ids = model.gpt.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_length_decoder,
                pad_token_id=tokenizer_target.pad_token_id,
                eos_token_id=tokenizer_target.eos_token_id,
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

            # Process labels
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
    bleu1, bleu2, bleu3, bleu4 = quick_bleu_metric(all_refs, all_preds, split=f'{log_what }Validation')
    
    # Save best model
    # Log metrics
    if log_what == "CISLR":
        if bleu4 > best_val_B4 or (bleu4 == best_val_B4 and avg_eval_loss < best_val_loss):
            best_val_B4 = bleu4
            best_val_loss = avg_eval_loss
            print('Saving CISLR best model checkpoint...')
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, best_val_B4, best_val_loss, best_checkpoint_path, 
                current_step, best_val_B4_isign, best_val_loss_isign, best_val_B1_isign, epoch_steps
            )
            
            df = pd.DataFrame({
                'Reference': [' '.join(ref[0]) for ref in all_refs],
                'Prediction': [' '.join(pred) for pred in all_preds]
            })
            df.to_csv(f'predictions_new/{project_name}_{sub_project_name}_predictions{log_what}.csv', index=False)
        
        wandb.log({
            'epoch': epoch + 1,
            'val/eval_loss': avg_eval_loss,
            'val/bleu1': bleu1 * 100,
            'val/bleu2': bleu2 * 100,
            'val/bleu3': bleu3 * 100,
            'val/bleu4': bleu4 * 100
        })
    elif log_what == "ISIGN":
        if counter >= 3:
            if bleu4 > best_val_B4_isign or (bleu4 == best_val_B4_isign and avg_eval_loss < best_val_loss_isign):
                best_val_B4_isign = bleu4
                best_val_loss_isign = avg_eval_loss
                print('Saving IsignB4 best model checkpoint...')
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch, best_val_B4, best_val_loss, best_checkpoint_path_isignB4, 
                    current_step, best_val_B4_isign, best_val_loss_isign, best_val_B1_isign, epoch_steps
                )
                
                df = pd.DataFrame({
                    'Reference': [' '.join(ref[0]) for ref in all_refs],
                    'Prediction': [' '.join(pred) for pred in all_preds]
                })
                df.to_csv(f'predictions_new/{project_name}_{sub_project_name}_predictions{log_what}B4.csv', index=False)
            
            if bleu1 > best_val_B1_isign or (bleu1 == best_val_B1_isign and avg_eval_loss < best_val_loss_isign):
                best_val_B1_isign = bleu1
                best_val_loss_isign = avg_eval_loss
                print('Saving IsignB1 best model checkpoint...')
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch, best_val_B4, best_val_loss, best_checkpoint_path_isignB1, 
                    current_step, best_val_B4_isign, best_val_loss_isign,best_val_B1_isign, epoch_steps
                )
                
                df = pd.DataFrame({
                    'Reference': [' '.join(ref[0]) for ref in all_refs],
                    'Prediction': [' '.join(pred) for pred in all_preds]
                })
                df.to_csv(f'predictions_new/{project_name}_{sub_project_name}_predictions{log_what}B1.csv', index=False)
            
        wandb.log({
            'val/eval_loss_isign': avg_eval_loss,
            'val/bleu1_isign': bleu1 * 100,
            'val/bleu2_isign': bleu2 * 100,
            'val/bleu3_isign': bleu3 * 100,
            'val/bleu4_isign': bleu4 * 100
        })
    
    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Resume training
    model.train()
    
    return best_val_B4, best_val_loss, best_val_B4_isign, best_val_B1_isign, best_val_loss_isign


# Start a timer
start_time = time.time()


# Step 7: Training and Evaluation Loop with BLEU Tracking
# Training and Evaluation Loop
for epoch in range(start_epoch, num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    counter = 0
    #progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")
    
    ## Hacky way to get infinite data.
    while (True):
        if epoch_steps % 1000 == 0:
            print(f"Training_step: {epoch_steps}")
        optimizer.zero_grad()
        threshold = get_threshold(epoch_steps, steps_for_100percentIsign)
        if random.random() < threshold:
            try:
                batch = next(isign_loader_cycle)
            except StopIteration:
                isign_loader_cycle = cycle(isign_loader)  # Restart the loader
                batch = next(isign_loader_cycle)
        else:
            try:
                batch = next(cislr_loader_cycle)
            except StopIteration:
                cislr_loader_cycle = cycle(train_loader)  # Restart the loader
                batch = next(cislr_loader_cycle)

        
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        pose_features = batch['pose_features'].to(device) 
        attention_mask = batch['attention_mask'].to(device)
        pose_mask = batch['pose_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            pose_features=pose_features, 
            attention_mask=attention_mask,
            pose_mask=pose_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) ,
            max_norm=1.0
        )
        # wandb.log({
        #     "learning_rate": scheduler.get_last_lr()[0],
        #     "step": epoch_steps
        # })
        

        optimizer.step()
        scheduler.step()
        
        epoch_steps += 1
        train_loss += loss.item()
        #progress_bar.set_postfix({'Loss': loss.item()})
        
        # Evaluation phase (every 1000 steps)
        if epoch_steps % 2500 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            counter += 1
            best_val_B4, best_val_loss, best_val_B4_isign, best_val_B1_isign, best_val_loss_isign = model_eval(
                eval2_loader, "ISIGN", best_val_B4,best_val_loss, best_val_B4_isign, 
                best_val_B1_isign, best_val_loss_isign,counter,  epoch_steps, epoch_steps, save_model=True)
        
            best_val_B4, best_val_loss, best_val_B4_isign, best_val_B1_isign, best_val_loss_isign = model_eval(
                eval_loader, "CISLR", best_val_B4, best_val_loss, best_val_B4_isign, 
                best_val_B1_isign, best_val_loss_isign, counter, epoch_steps, epoch_steps,save_model=True)
            
    # End of epoch
    avg_train_loss = train_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs} - Average Training Loss: {avg_train_loss:.4f}')
    current_lr = optimizer.param_groups[0]['lr']
    wandb.log({'epoch': epoch+1, 'train/train_loss': avg_train_loss, 'learning_rate': current_lr})
    
    # Save regular checkpoint
    save_checkpoint(
        model, feature_projection, optimizer, scheduler,
        epoch, best_val_B4, best_val_loss, checkpoint_path,epoch_steps, best_val_B4_isign, best_val_loss_isign, best_val_B1_isign
    )

# Finish the wandb run
wandb.finish()