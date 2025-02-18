import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

# Set the visible GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Configuration
vocab_size_decoder = 15000
dropout = 0.1
max_length_decoder = 128
decoder_hidden_size = 512
num_attention_heads = 8
num_decoder_layers = 4 
num_epochs = 1
learning_rate = 5e-5

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=max_length_decoder):
        self.tokenizer = tokenizer
        self.encodings = []
        # Process each text separately using encode
        for text in tqdm(texts, desc="Encoding texts"):
            try:
                # Encode the text
                text = f"<s> {text} </s>"
                encoding = tokenizer.encode(text)
                
                # Get the ids and attention mask
                input_ids = encoding.ids
                attention_mask = encoding.attention_mask
                
                # Always pad/truncate to max_length
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]
                    attention_mask = attention_mask[:max_length]
                else:
                    # Calculate padding length
                    pad_length = max_length - len(input_ids)
                    # Pad with pad token ID
                    pad_token_id = tokenizer.token_to_id("<pad>")
                    input_ids.extend([pad_token_id] * pad_length)
                    attention_mask.extend([0] * pad_length)
                
                # Convert to tensors of fixed size
                self.encodings.append({
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
                })
            except Exception as e:
                print(f"Error processing text: {text[:50]}... Error: {str(e)}")
                continue

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings[idx]['input_ids'],
            'attention_mask': self.encodings[idx]['attention_mask']
        }
        # Shift labels to left by 1 and add padding token at the end
        labels = item['input_ids'].clone()
        labels = torch.cat([labels[1:], torch.tensor([self.tokenizer.token_to_id("<pad>")])])
        item['labels'] = labels
        return item

    def __len__(self):
        return len(self.encodings)    

def train_tokenizer_from_scratch(texts, vocab_size=vocab_size_decoder, min_frequency=2):
    tokenizer = ByteLevelBPETokenizer(dropout=None, unicode_normalizer="nfc")
    
    with open("temp_train.txt", "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")
    
    # Train the tokenizer
    tokenizer.train(
        files=["temp_train.txt"],
        vocab_size=vocab_size - 7,
        min_frequency=min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>", "<PERSON>", "<UNKNOWN>"]
    )
    
    # Get the vocabulary
    vocab = tokenizer.get_vocab()
    
    # Create a new vocabulary with special tokens at the start
    special_tokens_dict = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "<PERSON>": 5,
        "<UNKNOWN>": 6
    }
    
    # Create new vocabulary dictionary
    new_vocab = {}
    for token, id in special_tokens_dict.items():
        new_vocab[token] = id
    
    # Add remaining tokens
    current_id = len(special_tokens_dict)
    for token, _ in vocab.items():
        if token not in special_tokens_dict:
            new_vocab[token] = current_id
            current_id += 1
    
    # Update tokenizer's vocabulary
    tokenizer.vocab = new_vocab
    tokenizer.ids_to_tokens = {v: k for k, v in new_vocab.items()}
    
    os.remove("temp_train.txt")
    
    # Verify special tokens
    def verify_special_tokens(tokenizer):
        for token, expected_id in special_tokens_dict.items():
            encoded = tokenizer.encode(token).ids
            if len(encoded) != 1 or encoded[0] != expected_id:
                print(f"Warning: {token} - Expected ID: {expected_id}, Got: {encoded}")
    
    verify_special_tokens(tokenizer)
    
    os.makedirs("custom_tokenizer", exist_ok=True)
    tokenizer.save_model("custom_tokenizer")
    
    return tokenizer

def create_custom_gpt2_config(tokenizer_target):
    """Create a custom GPT-2 configuration"""
    vocab = tokenizer_target.get_vocab()
    
    return GPT2Config(
        vocab_size=tokenizer_target.get_vocab_size(),
        n_positions=max_length_decoder,
        n_embd=decoder_hidden_size,
        n_layer=num_decoder_layers,
        n_head=num_attention_heads,
        pad_token_id=vocab.get("<pad>"),
        bos_token_id=vocab.get("<s>"),
        eos_token_id=vocab.get("</s>"),
        decoder_start_token_id = vocab.get("<s>"),
        add_cross_attention=True,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        resid_pdrop=dropout
    )
def calculate_perplexity(model, eval_dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Calculating perplexity"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get number of tokens excluding padding
            num_tokens = batch['attention_mask'].sum().item()
            
            outputs = model(**batch)
            loss = outputs.loss
            
            # Accumulate loss and token count
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Calculate average negative log likelihood
    avg_nll = total_loss / total_tokens
    
    # Calculate perplexity
    perplexity = np.exp(avg_nll)
    
    return perplexity

def train_model(model, train_dataloader, val_dataloader, device, num_epochs=num_epochs):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    best_val_loss = float('inf')
    best_perplexity = float('inf')
    
    # Lists to store metrics for plotting
    train_perplexities = []
    val_perplexities = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix({'train_loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Calculate training perplexity
        train_perplexity = calculate_perplexity(model, train_dataloader, device)
        train_perplexities.append(train_perplexity)
        
        # Validation phase
        model.eval()
        val_loss = 0
        progress_bar = tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}")
        
        with torch.no_grad():
            for batch in progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                val_loss += loss.item()
                progress_bar.set_postfix({'val_loss': loss.item()})
        
        avg_val_loss = val_loss / len(val_dataloader)
        
        # Calculate validation perplexity
        val_perplexity = calculate_perplexity(model, val_dataloader, device)
        val_perplexities.append(val_perplexity)
        
        print(f"Epoch {epoch + 1} metrics:")
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Training perplexity: {train_perplexity:.4f}")
        print(f"Average validation loss: {avg_val_loss:.4f}")
        print(f"Validation perplexity: {val_perplexity:.4f}")
        
        # Save best model based on validation perplexity
        if val_perplexity < best_perplexity:
            best_perplexity = val_perplexity
            print(f"New best validation perplexity: {best_perplexity:.4f}")
            model.save_pretrained('custom_gpt2_how2sign/best_model')
    
    # Save perplexity metrics
    metrics = {
        'train_perplexity': train_perplexities,
        'val_perplexity': val_perplexities
    }
    torch.save(metrics, 'custom_gpt2_how2sign/training_metrics.pt')
    
    return model

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    df1 = pd.read_csv('/DATA3/vaibhav/isign/PretrainingISL/train_MT16M.csv')
    df2 = pd.read_csv('/DATA3/vaibhav/isign/PretrainingISL/isign_new.csv')
    texts = df1['text'].tolist() + df2['text'].tolist()
    
    # Split into train and validation sets
    train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)
    print(f"Training on {len(train_texts)} texts, validating on {len(val_texts)} texts")
    
    # Train tokenizer from scratch or load existing
    print("Training/loading tokenizer...")
    tokenizer = train_tokenizer_from_scratch(train_texts)
    
    # Create custom configuration
    print("Creating model configuration...")
    config = create_custom_gpt2_config(tokenizer)
    
    # Initialize model with custom config
    print("Initializing model...")
    model = GPT2LMHeadModel(config)
    model.to(device)
    
    # Create datasets and dataloaders
    print("Preparing datasets...")
    train_dataset = TextDataset(train_texts, tokenizer)
    val_dataset = TextDataset(val_texts, tokenizer)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    # Train the model
    print("Starting training...")
    trained_model = train_model(model, train_dataloader, val_dataloader, device)
    
    # Calculate final perplexity
    final_perplexity = calculate_perplexity(trained_model, val_dataloader, device)
    print(f"Final validation perplexity: {final_perplexity:.4f}")
        
    # Save the final model and tokenizer
    print("Saving final model and tokenizer...")
    trained_model.save_pretrained('custom_gpt2/final_model')
    tokenizer.save_model('custom_gpt2')
    
    print("Training complete!")

if __name__ == "__main__":
    main()