import torch
import torch.nn as nn
import torch.optim as optim
from nn_data_gen import DataGenerator
from nn_model_v3 import MetaTransformerCracker
from nn_model import JamoTokenizer
import sys
import io

# Force UTF-8
if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def train_v3():
    gen = DataGenerator("corpus.txt")
    jamo_chars = [chr(i) for i in range(0x3131, 0x314f)] + [chr(i) for i in range(0x314f, 0x3164)] + [' ']
    tokenizer = JamoTokenizer(jamo_chars)
    
    vocab_size = tokenizer.vocab_size
    d_model = 256
    nhead = 4
    num_layers = 4
    batch_size = 64
    seq_len = 40
    epochs = 3000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetaTransformerCracker(vocab_size, d_model, nhead, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char2idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"Training V3 (Meta-Transformer) on {device}...")
    
    for epoch in range(epochs):
        model.train()
        batch = gen.generate_batch(batch_size, seq_len)
        
        src_batch = [tokenizer.encode(c) for c, p in batch]
        tgt_batch = [tokenizer.encode(p) for c, p in batch]
            
        src = torch.tensor(src_batch).transpose(0, 1).to(device) # (S, B)
        tgt = torch.tensor(tgt_batch).transpose(0, 1).to(device) # (S, B)
        
        optimizer.zero_grad()
        output = model(src)
        
        loss = criterion(output.view(-1, vocab_size), tgt.reshape(-1))
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            model.eval()
            with torch.no_grad():
                val_c, val_p = batch[0]
                val_src = torch.tensor(tokenizer.encode(val_c)).unsqueeze(1).to(device)
                val_out = model(val_src)
                pred_indices = val_out.argmax(dim=2).squeeze().tolist()
                pred_text = tokenizer.decode(pred_indices)
                print(f"  Orig: {val_p}")
                print(f"  Pred: {pred_text}")

    torch.save(model.state_dict(), "jamo_nn_v3.pth")
    print("V3 Model saved.")

if __name__ == "__main__":
    train_v3()
