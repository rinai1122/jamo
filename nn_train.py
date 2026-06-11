import torch
import torch.nn as nn
import torch.optim as optim
from nn_data_gen import DataGenerator
from nn_model import TransformerModel, JamoTokenizer
import sys
import io

# Force UTF-8
if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def train():
    # Setup data
    gen = DataGenerator("corpus.txt")
    jamo_chars = [chr(i) for i in range(0x3131, 0x314f)] + [chr(i) for i in range(0x314f, 0x3164)] + [' ']
    tokenizer = JamoTokenizer(jamo_chars)
    
    # Hyperparameters
    vocab_size = tokenizer.vocab_size
    d_model = 128
    nhead = 4
    num_layers = 3
    batch_size = 64
    seq_len = 40
    epochs = 1000 # Small epochs for demo, should be more for real training
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(vocab_size, d_model, nhead, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"Training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        batch = gen.generate_batch(batch_size, seq_len)
        
        # Prepare tensors
        src_batch = []
        tgt_batch = []
        for c, p in batch:
            src_batch.append(tokenizer.encode(c))
            tgt_batch.append(tokenizer.encode(p))
            
        src = torch.tensor(src_batch).transpose(0, 1).to(device) # (seq_len, batch_size)
        tgt = torch.tensor(tgt_batch).transpose(0, 1).to(device) # (seq_len, batch_size)
        
        optimizer.zero_grad()
        output = model(src) # (seq_len, batch_size, vocab_size)
        
        # Flatten for loss
        loss = criterion(output.view(-1, vocab_size), tgt.reshape(-1))
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            # Simple validation check
            model.eval()
            with torch.no_grad():
                val_c, val_p = batch[0]
                val_src = torch.tensor(tokenizer.encode(val_c)).unsqueeze(1).to(device)
                val_out = model(val_src)
                pred_indices = val_out.argmax(dim=2).squeeze().tolist()
                pred_text = tokenizer.decode(pred_indices)
                print(f"  Sample Original:  {val_p}")
                print(f"  Sample Predicted: {pred_text}")

    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': tokenizer.vocab
    }, "jamo_nn_model.pth")
    print("Model saved to jamo_nn_model.pth")

if __name__ == "__main__":
    train()
