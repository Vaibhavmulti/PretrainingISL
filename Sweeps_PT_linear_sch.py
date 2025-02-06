project_name = "CISLR_Pretraining_Sweeps_New"
sub_project_name = "Linear_schedulePT1"
import pandas as pd
import re
import os
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

from tqdm import tqdm
from transformers import (
    BertConfig, BertModel,
    GPT2Config, GPT2LMHeadModel,
    EncoderDecoderModel,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    get_linear_schedule_with_warmup
)
from datasets import Dataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from process_data_kmeansCISLR import get_pose_keypoints
from helpers.bleu_cal import quick_bleu_metric
from helpers.dataloaders import FeatureVectorDataset, FeatureVectorDataset_Isign
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer


#'name': 'val/bleu4'


sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val/bleu4',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'min': 1e-5,
            'max': 9e-4,
            'distribution': 'log_uniform'
        },
        'weight_decay': { 
            'min': 1e-6,
            'max': 1e-4,
            'distribution': 'log_uniform'
        },
        'batch_size': {
            'values': [16, 64, 128]
        },
        'num_encoder_layers': {
            'values': [2, 4, 6, 8]
        },
        'num_decoder_layers': {
            'values': [2, 4, 6, 8]
        },
        'encoder_hidden_size': {
            'values': [256, 512, 768]
        },
        'decoder_hidden_size': {
            'values': [256, 512, 768]
        },
        'num_attention_heads': {
            'values': [4, 8]
        },
        'dropout': {
            'min': 0.1,
            'max': 0.3,
            'distribution': 'uniform'
        },
        'warmup_ratio': {
            'min': 0.05,
            'max': 0.2,
            'distribution': 'uniform'
        },
        'num_beams': {
            'values': [2, 3, 4, 5]
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3
    }
}


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set random seed for reproducibility
set_seed()


#Hyperparameters here now 
# learning_rate = 3e-4 #Slower learning rate for finetuning
# num_encoder_layers = 4
# num_decoder_layers = 4
# encoder_hidden_size = 512
# decoder_hidden_size = 512
# num_attention_heads = 8
# dropout = 0.1
# num_beams = 3
# batch_size = 16 #256

num_epochs = 1
MAX_FRAMES = 300  # Max video frames.
max_position_embeddings_encoder = MAX_FRAMES
max_length_decoder = 128
vocab_size_decoder = 15000
num_keypoints = 152 # We have cherrypicked these
POSE_DIR = "/DATA1007/vaibhav/tokenization/CISLR/CISLR_v1.5-a_videos_poses/"
POSE_DIR_ISIGN = "/DATA1007/vaibhav/isign/Data/iSign-poses_v1.1/"
randomize_word_order = False



# # Set the visible GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "2"



train_df = pd.read_csv('/DATA1007/vaibhav/tokenization/CISLR/train.csv')
eval_df = pd.read_csv('/DATA1007/vaibhav/tokenization/CISLR/val.csv')
test_df = pd.read_csv('/DATA1007/vaibhav/tokenization/CISLR/test.csv')


eval_df2 = pd.read_csv('/DATA1007/vaibhav/tokenization/val_split_unicode_filtered.csv')
train_df2 = pd.read_csv('/DATA1007/vaibhav/tokenization/train_split_unicode_filtered.csv')

# Step 2: Train Tokenizers
# Combine source and target sequences for a joint tokenizer
#all_sequences = train_df['SENTENCE_UNICIDE'].tolist() + train_df['text'].tolist()
    
all_sequences_target = train_df['text'].tolist() + train_df2['text'].tolist()


# # Initialize and train the tokenizer
# tokenizer_model = models.BPE()
# tokenizer = Tokenizer(tokenizer_model)
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# trainer = trainers.BpeTrainer(
#     vocab_size=vocab_size_decoder,
#     special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
# )

# tokenizer.train_from_iterator(all_sequences_target, trainer=trainer)
# # Save the tokenizer
# #Make tokenizer_file if it does not exist

# if not os.path.exists('tokenizer_file'):
#     os.makedirs('tokenizer_file')

# tokenizer.save("tokenizer_file/target_tokenizer.json")

# Load the tokenizer as a PreTrainedTokenizerFast
tokenizer_target = PreTrainedTokenizerFast(tokenizer_file="tokenizer_file/target_tokenizer.json")
tokenizer_target.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>"
})



# Extract video UIDs and labels

print('Extracting video UIDs and labels...')
train_video_uids = train_df['uid_list'].apply(ast.literal_eval).tolist()
eval_video_uids = eval_df['uid_list'].apply(ast.literal_eval).tolist()
test_video_uids = test_df['uid_list'].apply(ast.literal_eval).tolist()


eval2_video_uids = eval_df2['uid'].tolist()

print('Appending <s> and </s> to labels...')
train_labels = [f'<s>{text}</s>' for text in train_df['text'].tolist()]
eval_labels = [f'<s>{text}</s>' for text in eval_df['text'].tolist()]
test_labels = [f'<s>{text}</s>' for text in test_df['text'].tolist()]

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


eval2_labels = tokenizer_target(eval2_labels, max_length=max_length_decoder, padding="max_length", truncation=True)['input_ids']

# Create datasets
print('Creating datasets...')
train_dataset = FeatureVectorDataset(train_video_uids,tokenizer_target, randomize_word_order, train_labels)
test_dataset = FeatureVectorDataset(test_video_uids,tokenizer_target, randomize_word_order, test_labels)
eval_dataset = FeatureVectorDataset(eval_video_uids,tokenizer_target, randomize_word_order,eval_labels)

eval2_dataset = FeatureVectorDataset_Isign(eval2_video_uids, tokenizer_target, eval2_labels)

print('Creating DataLoaders...')
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
# eval_loader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=2)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=2)

# eval2_loader = DataLoader(eval2_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=2)


# Step 4: Define the Encoder and Decoder Models

# Linear layer to project feature vectors to the expected input shape
class FeatureProjection(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureProjection, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)



# Load the best model checkpoint if it exists
# Create directory if it does not exist
if not os.path.exists('load_pretrained'):
    os.makedirs('load_pretrained')

if not os.path.exists('predictions_new'):
    os.makedirs('predictions_new')

#load_path = "load_pretrained/"+project_name+'_'+sub_project_name+'_'+"best_model_checkpoint.pth"
#load_path = "/DATA1007/vaibhav/tokenization/CISLR/predictions/CISLR_Pretraining_Pretraining1_best_model_checkpoint.pth"
#load_path = "/DATA1007/vaibhav/tokenization/CISLR/predictions_new/CISLR_Pretraining_ScratchPretraining1_commonvocab_best_model_checkpoint.pth"
load_path = ""
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



def model_eval(model,feature_projection,epoch, encoder_config,optimizer, scheduler,num_beams, eval_loader, log_what, best_val_B4,save_model=False):
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
    bleu1, bleu2, bleu3, bleu4 = quick_bleu_metric(all_refs, all_preds, split=f'{log_what }Validation')
    
    # Save best model
    if save_model:    
        if bleu4 > best_val_B4 or (bleu4 == best_val_B4 and avg_eval_loss < best_val_loss):
            best_val_B4 = bleu4
            best_val_loss = avg_eval_loss
            print('Saving best model checkpoint...')
            save_checkpoint(
                model, feature_projection, optimizer, scheduler,
                epoch, best_val_B4, best_val_loss, best_checkpoint_path
            )
            
            df = pd.DataFrame({
                'Reference': [' '.join(ref[0]) for ref in all_refs],
                'Prediction': [' '.join(pred) for pred in all_preds]
            })
            df.to_csv(f'predictions_new/{project_name}_{sub_project_name}_predictions{log_what}.csv', index=False)
        
    
    # Log metrics
    if log_what == "CISLR":
        wandb.log({
            'epoch': epoch + 1,
            'val/eval_loss': avg_eval_loss,
            'val/bleu1': bleu1 * 100,
            'val/bleu2': bleu2 * 100,
            'val/bleu3': bleu3 * 100,
            'val/bleu4': bleu4 * 100
        })
    elif log_what == "ISIGN":
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
    feature_projection.train()


# Step 7: Training and Evaluation Loop with BLEU Tracking
# Training and Evaluation Loop

def main():
    # Initialize wandb sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    
    def train_sweep():
        # Initialize new wandb run
        run = wandb.init(project=project_name, name=sub_project_name)
        config = wandb.config
        
        # Encoder Configuration and Model
        encoder_config = BertConfig(
            hidden_size=config.encoder_hidden_size,
            num_hidden_layers=config.num_encoder_layers,
            num_attention_heads=config.num_attention_heads,
            hidden_dropout_prob=config.dropout,  # Dropout after fully connected layers
            attention_probs_dropout_prob=config.dropout,  # Dropout on attention weights
        )
        encoder = BertModel(encoder_config)
        print(encoder_config)

        # Decoder Configuration and Model
        decoder_config = GPT2Config(
            vocab_size=tokenizer_target.vocab_size,
            n_positions=max_length_decoder, # We have padded and truncated to 128
            n_embd=config.decoder_hidden_size,
            n_layer=config.num_decoder_layers,
            n_head=config.num_attention_heads,
            pad_token_id=tokenizer_target.pad_token_id,
            bos_token_id=tokenizer_target.bos_token_id,
            eos_token_id=tokenizer_target.eos_token_id,
            add_cross_attention=True,  # Important for Seq2Seq models (Can't find this on HF docs)
            embd_pdrop=config.dropout,  # Dropout on embeddings 
            attn_pdrop=config.dropout,  # Dropout on attention probabilities 
            resid_pdrop=config.dropout  # Dropout on residual connections 
        )
        print(decoder_config)
        decoder = GPT2LMHeadModel(decoder_config)

        # Combine Encoder and Decoder into EncoderDecoderModel
        feature_projection = FeatureProjection(num_keypoints, encoder_config.hidden_size)
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

        # Tie weights (optional)
        model.config.decoder_start_token_id = tokenizer_target.bos_token_id
        model.config.eos_token_id = tokenizer_target.eos_token_id
        model.config.pad_token_id = tokenizer_target.pad_token_id
        model.config.vocab_size = decoder_config.vocab_size
        model.config.max_length = max_length_decoder

        # Initialize model, optimizer, and scheduler
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        feature_projection.to(device)
    
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(feature_projection.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Update dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2
        )
        
        # Scheduler setup
        total_steps = len(train_loader)
        warmup_steps = int(config.warmup_ratio * total_steps)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        start_epoch = 0
        best_val_B4 = 0.0
        best_val_loss = float('inf')

        # Existing training loop
        for epoch in range(start_epoch, num_epochs):
            # Training phase
            model.train()
            feature_projection.train()
            train_loss = 0.0
            epoch_steps = 0
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
                    "step": epoch_steps
                }, step=epoch_steps)
                

                optimizer.step()
                scheduler.step()
                
                epoch_steps += 1
                train_loss += loss.item()
                progress_bar.set_postfix({'Loss': loss.item()})
                
                # Evaluation phase (every 1000 steps)
                if epoch_steps % 2500 == 0:
                    model_eval(model, feature_projection, epoch, encoder_config,optimizer, scheduler,config.num_beams, eval_loader, "CISLR", best_val_B4, save_model=False)
                    #model_eval(eval2_loader, "ISIGN", best_val_B4,  save_model=False)
                # if epoch_steps % 7500 == 0:
                #     # Save regular checkpoint
                #     save_checkpoint(
                #         model, feature_projection, optimizer, scheduler,
                #         epoch, best_val_B4, best_val_loss,
                #         "predictions_new/"+project_name+'_'+sub_project_name+'_'+str(epoch_steps)+"checkpoint.pth"
                #     )

            # End of epoch
            avg_train_loss = train_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{num_epochs} - Average Training Loss: {avg_train_loss:.4f}')
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({'epoch': epoch+1, 'train/train_loss': avg_train_loss, 'learning_rate': current_lr})
            
            # Save regular checkpoint
            save_checkpoint(
                model, feature_projection, optimizer, scheduler,
                epoch, best_val_B4, best_val_loss, checkpoint_path
            )
    
    # Run sweep
    wandb.agent(sweep_id, train_sweep, count=20)

if __name__ == "__main__":
    main()