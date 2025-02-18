import torch
from transformers import GPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer

def load_trained_model(model_path='custom_gpt2/best_model', tokenizer_path='custom_tokenizer'):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the tokenizer
    try:
        tokenizer = ByteLevelBPETokenizer(
            f"{tokenizer_path}/vocab.json",
            f"{tokenizer_path}/merges.txt"
        )
        print("Tokenizer loaded successfully")
    except Exception as e:
        raise Exception(f"Error loading tokenizer: {str(e)}")
    
    # Load the model
    try:
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.to(device)
        model.eval()  # Set to evaluation mode
        print("Model loaded successfully")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
    
    return model, tokenizer

# Example usage:
if __name__ == "__main__":
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, tokenizer = load_trained_model()
    
    # Example text for inference
    text = "Congress"
    
    # Encode text
    encoding = tokenizer.encode(text)
    input_ids = torch.tensor([encoding.ids]).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            pad_token_id=tokenizer.token_to_id("<pad>"),
            eos_token_id=tokenizer.token_to_id("</s>"),
            do_sample=True,
            temperature=0.7
        )
    
    # Decode and print result
    for output in outputs:
        decoded = tokenizer.decode(output.tolist())
        print(f"Generated: {decoded}")