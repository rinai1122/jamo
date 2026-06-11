import torch
import torch.nn as nn
import torch.optim as optim
from nn_data_gen import DataGenerator
from nn_model_v2 import TransformerCracker
from nn_model import JamoTokenizer
import sys
import io
import time

# Force UTF-8
if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)

def train_v2():
    # Setup data
    gen = DataGenerator("corpus.txt")
    jamo_chars = [chr(i) for i in range(0x3131, 0x314f)] + [chr(i) for i in range(0x314f, 0x3164)] + [' ']
    tokenizer = JamoTokenizer(jamo_chars)
    
    # Hyperparameters
    vocab_size = tokenizer.vocab_size
    d_model = 256
    nhead = 4
    num_layers = 3
    batch_size = 32
    seq_len = 40
    epochs = 2000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerCracker(vocab_size, d_model, nhead, num_layers, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char2idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"Training V2 on {device}...")
    
    for epoch in range(epochs):
        model.train()
        batch = gen.generate_batch(batch_size, seq_len)
        
        # Prepare tensors
        src_batch = []
        tgt_batch_input = []
        tgt_batch_expected = []
        
        sos_idx = tokenizer.char2idx['<SOS>']
        eos_idx = tokenizer.char2idx['<EOS>']
        
        for c, p in batch:
            src_batch.append(tokenizer.encode(c))
            p_encoded = tokenizer.encode(p)
            tgt_batch_input.append([sos_idx] + p_encoded)
            tgt_batch_expected.append(p_encoded + [eos_idx])
            
        src = torch.tensor(src_batch).transpose(0, 1).to(device) # (S, B)
        tgt_in = torch.tensor(tgt_batch_input).transpose(0, 1).to(device) # (T, B)
        tgt_exp = torch.tensor(tgt_batch_expected).transpose(0, 1).to(device) # (T, B)
        
        tgt_mask = generate_square_subsequent_mask(tgt_in.size(0), device)
        
        optimizer.zero_grad()
        output = model(src, tgt_in, tgt_mask=tgt_mask)
        
        loss = criterion(output.view(-1, vocab_size), tgt_exp.reshape(-1))
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            # Inference check
            model.eval()
            with torch.no_grad():
                val_c, val_p = batch[0]
                val_src = torch.tensor(tokenizer.encode(val_c)).unsqueeze(1).to(device)
                
                # Simple greedy decoding
                decoded_indices = [sos_idx]
                for _ in range(seq_len):
                    curr_tgt = torch.tensor(decoded_indices).unsqueeze(1).to(device)
                    t_mask = generate_square_subsequent_mask(curr_tgt.size(0), device)
                    out = model(val_src, curr_tgt, tgt_mask=t_mask)
                    next_idx = out[-1].argmax().item()
                    decoded_indices.append(next_idx)
                    if next_idx == eos_idx: break
                
                pred_text = tokenizer.decode(decoded_indices[1:-1])
                print(f"  Orig: {val_p}")
                print(f"  Pred: {pred_text}")

    torch.save(model.state_dict(), "jamo_nn_v2.pth")
    print("V2 Model saved.")

if __name__ == "__main__":
    train_v2()
