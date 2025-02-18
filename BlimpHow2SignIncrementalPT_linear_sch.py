# Uncomment bestisgnb4path if you want to load from the checkpoint.
# Threshold 
project_name = "How2SignPretrain"
sub_project_name = "BLIMPH2S1.1M_paramsPT2"
run_name = "BLIMPH2S1.1M_paramsPT2"

# Gausian Noise , Random frame sampling , Isign mixed with CISLR linearly

randomize_word_order = True
steps_for_100percentIsign = 60000
import os
import pandas as pd
# # Set the visible GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "2"




train_df = pd.read_csv('/DATA3/vaibhav/isign/PretrainingISL/train_BLIMPWASL.csv')
eval_df = pd.read_csv('/DATA3/vaibhav/isign/PretrainingISL/val_BLIMPWASL.csv')
test_df = pd.read_csv('/DATA3/vaibhav/isign/PretrainingISL/test_BLIMPWASL.csv')

# train_df = train_df.sample(n=1000)
# eval_df = eval_df.sample(n=1000)


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
import sacrebleu

from tqdm import tqdm
from transformers import (
    BertConfig, BertModel,
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    EncoderDecoderModel,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    get_constant_schedule_with_warmup
)
from datasets import Dataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from helpers.bleu_cal import quick_bleu_metric
from helpers.dataloaders import FeatureVectorDataset, FeatureVectorDataset_Isign
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
    if total_steps == 0:
        return 1.1
    else:
        return min(current_step / total_steps, 0.85)

#Hyperparameters here now 
learning_rate = 1e-4 #3e-4 
num_encoder_layers = 2 #4
num_decoder_layers = 2 #4
encoder_hidden_size = 512 #512
decoder_hidden_size = 512 #512
num_attention_heads = 4 #8
dropout = 0.3 #0.1
MAX_FRAMES = 300  # Max video frames.
max_position_embeddings_encoder = MAX_FRAMES
num_beams = 3
#label_smoothing = 0.1 not used yet
warmup_steps_ratio = 0.1
batch_size = 16 #64 #256
#gradient_accumulation_steps = 1 not used yet
lr_scheduler_type = 'warmup_linear_constant_afterwards'
num_epochs = 1
max_length_decoder = 128
vocab_size_decoder = 15000
num_keypoints = 152 # We have cherrypicked these
WEIGTH_DECAY = 0.01

POSE_DIR = "/DATA7/sanjeet/ISL/WLASL/start_kit/pose_video/"
POSE_DIR_ISIGN = "/DATA7/sanjeet/ISL/WLASL/How2Sign/videos/How2Sign_pose_all/"
#STEP_FRAMES = None  # Random sampling of frames.
STEP_FRAMES_ISIGN = None
STEP_FRAMES_CISLR = 3
ADD_NOISE_ISIGN = False
ADD_NOISE_CISLR = True


hyperparameters = {'learning_rate': learning_rate, 
                     'num_encoder_layers': num_encoder_layers,
                        'num_decoder_layers': num_decoder_layers,
                        'encoder_hidden_size': encoder_hidden_size,
                        'decoder_hidden_size': decoder_hidden_size,
                        'num_attention_heads': num_attention_heads,
                        'dropout': dropout,
                        'max_position_embeddings_encoder': max_position_embeddings_encoder,
                        'num_beams': num_beams,
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
                        ,'sub_project_name': sub_project_name,
                        'steps_for_100percentIsign': steps_for_100percentIsign,
                        'ADD_NOISE_ISIGN': ADD_NOISE_ISIGN,
                        'ADD_NOISE_CISLR': ADD_NOISE_CISLR,
                        'STEP_FRAMES_ISIGN': STEP_FRAMES_ISIGN,
                        'STEP_FRAMES_CISLR': STEP_FRAMES_CISLR
                        }





wandb.init(project=project_name, name=run_name, config = hyperparameters)

#wandb.init(project=project_name, config = hyperparameters, id="c43wvsr1", resume="must")

#wandb.init(project=project_name, config = hyperparameters, id="7ike4lk8", resume="must")


eval_df2 = pd.read_csv('/DATA7/sanjeet/ISL/WLASL/How2Sign/how2sign_realigned_val_filtered1.csv', sep='\t')
eval_df2 = eval_df2.rename(columns={'SENTENCE': 'text', 'SENTENCE_ID': 'uid'})
#'/DATACSEShare/sanjeet/Dataset/Sign_lanuguage_data_set/isign/Final_Processed_raw_sentences_isign.csv'
#train_df2 = pd.read_csv('/DATA7/vaibhav/tokenization/train_split_unicode_filtered.csv')
print(eval_df2.columns)
train_df2 = pd.read_csv('/DATA3/vaibhav/isign/PretrainingISL/How2sign_train_with_without_punctuation.csv')
train_df2 = train_df2.rename(columns={'sentence_with_punctuation': 'text', 'SENTENCE_ID': 'uid'})
print(train_df2.columns)

# Step 2: Train Tokenizers
# Combine source and target sequences for a joint tokenizer
#all_sequences = train_df['SENTENCE_UNICIDE'].tolist() + train_df['text'].tolist()
# train_df2 = train_df2.sample(n=1000)
# eval_df2 = eval_df2.sample(n=1000)


all_sequences_target = train_df['text'].tolist() + train_df2['text'].tolist()
#all_sequences_target = train_df['text'].values.tolist() + train_df2['text'].values.tolist()


# from transformers import GPT2TokenizerFast

# tokenizer_target = GPT2TokenizerFast.from_pretrained('gpt2', 
#     bos_token="<s>",
#     eos_token="</s>",
#     unk_token="<unk>", 
#     pad_token="<pad>",
#     mask_token="<mask>",
#     additional_special_tokens=['<PERSON>', '<UNKNOWN>']
# )

#tokenizer_target = GPT2Tokenizer.from_pretrained('gpt2')
# tokenizer_target.pad_token = tokenizer_target.eos_token
# # Define special tokens

# print(len(tokenizer_target))
# tokenizer_target.add_special_tokens({
#     "bos_token": "<s>",
#     "eos_token": "</s>",
#     "unk_token": "<unk>",
#     "pad_token": "<pad>",
#     "mask_token": "<mask>",
#     'additional_special_tokens': ['<PERSON>', '<UNKNOWN>']
# })

# # Initialize and train the tokenizer
tokenizer_model = models.BPE()
tokenizer = Tokenizer(tokenizer_model)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size_decoder,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>", "<PERSON>", "<UNKNOWN>"]
)

tokenizer.train_from_iterator(all_sequences_target, trainer=trainer)
# Save the tokenizer
#Make tokenizer_file if it does not exist

if not os.path.exists('tokenizer_file'):
    os.makedirs('tokenizer_file')

tokenizer.save("tokenizer_file/target_tokenizer_h2s.json")

##Load the tokenizer as a PreTrainedTokenizerFast
tokenizer_target = PreTrainedTokenizerFast(tokenizer_file="tokenizer_file/target_tokenizer_h2s.json")
tokenizer_target.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    'additional_special_tokens': ['<PERSON>', '<UNKNOWN>']
})



# Extract video UIDs and labels

print('Extracting video UIDs and labels...')
train_video_uids = train_df['uid_list'].apply(ast.literal_eval).tolist()
eval_video_uids = eval_df['uid_list'].apply(ast.literal_eval).tolist()
test_video_uids = test_df['uid_list'].apply(ast.literal_eval).tolist()

train2_video_uids = train_df2['uid'].tolist()
eval2_video_uids = eval_df2['uid'].tolist()

print('Appending <s> and </s> to labels...')
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


train_dataset = FeatureVectorDataset(train_video_uids,tokenizer_target,
                                      randomize_word_order, MAX_FRAMES, POSE_DIR,
                                      train_labels, step_frames=STEP_FRAMES_CISLR, add_noise = ADD_NOISE_CISLR)
test_dataset = FeatureVectorDataset(test_video_uids,tokenizer_target, 
                                    randomize_word_order, MAX_FRAMES, POSE_DIR,
                                    test_labels,step_frames=STEP_FRAMES_CISLR, add_noise = ADD_NOISE_CISLR)
eval_dataset = FeatureVectorDataset(eval_video_uids,tokenizer_target, 
                                    randomize_word_order,MAX_FRAMES, POSE_DIR,
                                    eval_labels,step_frames=STEP_FRAMES_CISLR, add_noise = ADD_NOISE_CISLR)

train2_dataset = FeatureVectorDataset_Isign(train2_video_uids, tokenizer_target,
                                            MAX_FRAMES, POSE_DIR_ISIGN, train2_labels ,
                                           step_frames=STEP_FRAMES_ISIGN, add_noise = ADD_NOISE_ISIGN)

eval2_dataset = FeatureVectorDataset_Isign(eval2_video_uids, tokenizer_target, 
                                        MAX_FRAMES, POSE_DIR_ISIGN, eval2_labels, 
                                        step_frames=STEP_FRAMES_ISIGN, add_noise = ADD_NOISE_ISIGN)

# Create DataLoaders
print('Creating DataLoaders...')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size*4, num_workers=4, pin_memory=True, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size*4, num_workers=4, pin_memory=True, prefetch_factor=2)


isign_loader = DataLoader(train2_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
eval2_loader = DataLoader(eval2_dataset, batch_size=batch_size*4, num_workers=4, pin_memory=True, prefetch_factor=2)


isign_loader_cycle = cycle(isign_loader)  # To cycle through ISIGN when exhausted
cislr_loader_cycle = cycle(train_loader)  # To cycle through CISLR when exhausted



# Step 4: Define the Encoder and Decoder Models
# Encoder Configuration and Model
encoder_config = BertConfig(
    hidden_size=encoder_hidden_size,
    num_hidden_layers=num_encoder_layers,
    num_attention_heads=num_attention_heads,
    hidden_dropout_prob=dropout,  # Dropout after fully connected layers
    attention_probs_dropout_prob=dropout,  # Dropout on attention weights
)
#encoder = BertForCausalLM(encoder_config)
encoder = BertModel(encoder_config)
print(encoder_config)

# Decoder Configuration and Model
decoder_config = GPT2Config(
    vocab_size=len(tokenizer_target),
    n_positions=max_length_decoder, # We have padded and truncated to 128
    n_embd=decoder_hidden_size,
    n_layer=num_decoder_layers,
    n_head=num_attention_heads,
    pad_token_id=tokenizer_target.pad_token_id,
    bos_token_id=tokenizer_target.bos_token_id,
    eos_token_id=tokenizer_target.eos_token_id,
    add_cross_attention=True,  # Important for Seq2Seq models (Can't find this on HF docs)
    embd_pdrop=dropout,  # Dropout on embeddings 
    attn_pdrop=dropout,  # Dropout on attention probabilities 
    resid_pdrop=dropout  # Dropout on residual connections 
)
print(decoder_config)
decoder = GPT2LMHeadModel(decoder_config)
# decoder.resize_token_embeddings(len(tokenizer_target))
# Linear layer to project feature vectors to the expected input shape
class FeatureProjection(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims = 1024):
        super(FeatureProjection, self).__init__()
        # self.linear = torch.nn.Linear(input_dim, output_dim)
        self.linear1 = torch.nn.Linear(input_dim, hidden_dims)
        self.linear2 = torch.nn.Linear(hidden_dims, output_dim)
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        # return self.linear(x)
        x = self.gelu(self.linear1(x))
        x = self.linear2(x)
        return x


# Combine Encoder and Decoder into EncoderDecoderModel
feature_projection = FeatureProjection(num_keypoints, encoder_config.hidden_size)
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
#model.decoder.resize_token_embeddings(len(tokenizer_target))

# Tie weights (optional)
model.config.decoder_start_token_id = tokenizer_target.bos_token_id
model.config.eos_token_id = tokenizer_target.eos_token_id
model.config.pad_token_id = tokenizer_target.pad_token_id
model.config.vocab_size = decoder_config.vocab_size
model.config.max_length = max_length_decoder

# After creating your model and adding special tokens


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
def save_checkpoint(model, feature_projection, optimizer, scheduler, epoch, best_val_B4, best_val_loss, checkpoint_path, current_step,
                    best_val_B4_isign, best_val_loss_isign, best_val_B1_isign, epoch_steps):
    checkpoint = {
        'epoch': epoch,
        'current_step' : current_step,
        'model_state_dict': model.state_dict(),
        'feature_projection_state_dict': feature_projection.state_dict(),
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

def load_checkpoint(model, feature_projection, optimizer, scheduler, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        feature_projection.load_state_dict(checkpoint['feature_projection_state_dict'])
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
feature_projection.to(device)

optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(feature_projection.parameters()),
    weight_decay=WEIGTH_DECAY,
    lr=learning_rate
)

# Calculate total steps for scheduler
total_steps = len(train_loader)  
# Set warmup to 10% of total steps
warmup_steps = int(warmup_steps_ratio * total_steps)

# total_steps = len(train_loader) * num_epochs
#warmup_steps = len(train_loader) * warmup_steps_epocs

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
        model, feature_projection, optimizer, scheduler, best_checkpoint_path_isignB4
    )
    start_epoch = 0
    print("Loaded best model checkpoint IsignB4")
    print("*"*50)

elif os.path.exists(load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    feature_projection.load_state_dict(checkpoint['feature_projection_state_dict'])
    
    # Load optimizer and scheduler states but reset epoch counter
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
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
    feature_projection.eval()
    eval_loss = 0.0
    all_refs = []
    sacre_refs = []
    sacre_preds = []
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
            
            for ref in refs:
                sacre_refs.append(str(ref))
            
            for pred in preds:
                sacre_preds.append(str(pred))
            
            ref_tokens = [ref.strip().split() for ref in refs]
            pred_tokens = [pred.strip().split() for pred in preds]
            
            all_refs.extend([ref] for ref in ref_tokens)
            all_preds.extend(pred_tokens)
    
    # Calculate metrics
    avg_eval_loss = eval_loss / len(eval_loader)
    bleu1, bleu2, bleu3, bleu4 = quick_bleu_metric(all_refs, all_preds, split=f'{log_what }Validation')
    bleu_sacre = sacrebleu.corpus_bleu(sacre_preds, [sacre_refs])
    bleu_sacre1, bleu_sacre2, bleu_sacre3, bleu_sacre4 =  bleu_sacre.precisions[0], bleu_sacre.precisions[1], bleu_sacre.precisions[2], bleu_sacre.precisions[3]
    # Save best model
    # Log metrics
    if log_what == "WASL":

        print(f'Sacre Bleu1_WASL :{bleu_sacre1}')
        print(f'Sacre Bleu2_WASL :{bleu_sacre2}')
        print(f'Sacre Bleu3_WASL :{bleu_sacre3}')
        print(f'Sacre Bleu4_WASL :{bleu_sacre4}')
        if bleu4 > best_val_B4 or (bleu4 == best_val_B4 and avg_eval_loss < best_val_loss):
            best_val_B4 = bleu4
            best_val_loss = avg_eval_loss
            print('Saving WASL best model checkpoint...')
            save_checkpoint(
                model, feature_projection, optimizer, scheduler,
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
        wandb.log({
            'val/bleu1_sacre': bleu_sacre1,
            'val/bleu2_sacre': bleu_sacre2,
            'val/bleu3_sacre': bleu_sacre3,
            'val/bleu4_sacre': bleu_sacre4
        })
    elif log_what == "HOW2SIGN":
        print(f'Sacre Bleu1_HOW2SIGN :{bleu_sacre1}')
        print(f'Sacre Bleu2_HOW2SIGN :{bleu_sacre2}')
        print(f'Sacre Bleu3_HOW2SIGN :{bleu_sacre3}')
        print(f'Sacre Bleu4_HOW2SIGN :{bleu_sacre4}')
        if counter >= 1:
            if bleu4 > best_val_B4_isign or (bleu4 == best_val_B4_isign and avg_eval_loss < best_val_loss_isign):
                best_val_B4_isign = bleu4
                best_val_loss_isign = avg_eval_loss
                print('Saving HOW2SIGN best model checkpoint...')
                save_checkpoint(
                    model, feature_projection, optimizer, scheduler,
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
                print('Saving HOW2SIGNB1 best model checkpoint...')
                save_checkpoint(
                    model, feature_projection, optimizer, scheduler,
                    epoch, best_val_B4, best_val_loss, best_checkpoint_path_isignB1, 
                    current_step, best_val_B4_isign, best_val_loss_isign,best_val_B1_isign, epoch_steps
                )
                
                df = pd.DataFrame({
                    'Reference': [' '.join(ref[0]) for ref in all_refs],
                    'Prediction': [' '.join(pred) for pred in all_preds]
                })
                df.to_csv(f'predictions_new/{project_name}_{sub_project_name}_predictions{log_what}B1.csv', index=False)
            
        wandb.log({
            'val/eval_loss_HOW2SIGN': avg_eval_loss,
            'val/bleu1_HOW2SIGN': bleu1 * 100,
            'val/bleu2_HOW2SIGN': bleu2 * 100,
            'val/bleu3_HOW2SIGN': bleu3 * 100,
            'val/bleu4_HOW2SIGN': bleu4 * 100
        })
        wandb.log({
            'val/bleu1_sacre_HOW2SIGN': bleu_sacre1,
            'val/bleu2_sacre_HOW2SIGN': bleu_sacre2,
            'val/bleu3_sacre_HOW2SIGN': bleu_sacre3,
            'val/bleu4_sacre_HOW2SIGN': bleu_sacre4
        })
    
    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Resume training
    model.train()
    feature_projection.train()
    return best_val_B4, best_val_loss, best_val_B4_isign, best_val_B1_isign, best_val_loss_isign



# Step 7: Training and Evaluation Loop with BLEU Tracking
# Training and Evaluation Loop
for epoch in range(start_epoch, num_epochs):
    # Training phase
    model.train()
    feature_projection.train()
    train_loss = 0.0
    
    counter = 0
    #progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")
    
    ## Hacky way to get infinite data.
    while (True):
        if epoch_steps % 1000 == 0:
            print(f"Training_step: {epoch_steps}")
            print(f'Bett MT val B4: {best_val_B4}')
        optimizer.zero_grad()
        threshold = get_threshold(epoch_steps, steps_for_100percentIsign)
        #threshold = 1.1
        #threshold = best_val_B4
        
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
        })
        

        optimizer.step()
        scheduler.step()
        
        epoch_steps += 1
        train_loss += loss.item()
        #progress_bar.set_postfix({'Loss': loss.item()})
        
        # Evaluation phase (every 1000 steps)
        if epoch_steps % 2500 == 0:
            counter += 1
            best_val_B4, best_val_loss, best_val_B4_isign, best_val_B1_isign, best_val_loss_isign = model_eval(
                eval_loader, "WASL", best_val_B4, best_val_loss, best_val_B4_isign, 
                best_val_B1_isign, best_val_loss_isign, counter, epoch_steps, epoch_steps,save_model=True)
            best_val_B4, best_val_loss, best_val_B4_isign, best_val_B1_isign, best_val_loss_isign = model_eval(
                eval2_loader, "HOW2SIGN", best_val_B4,best_val_loss, best_val_B4_isign, 
                best_val_B1_isign, best_val_loss_isign,counter,  epoch_steps, epoch_steps, save_model=True)
        # if epoch_steps % 7500 == 0:
        #     # Save regular checkpoint
        #     save_checkpoint(
        #         model, feature_projection, optimizer, scheduler,
        #         epoch, best_val_B4, best_val_loss,
        #         "predictions_new/"+project_name+'_'+sub_project_name+'_'+str(epoch_steps)+"checkpoint.pth",
        #         epoch_steps
        #     )

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