from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.adam
from minbpe import RegexTokenizer

batch_size = 32
block_size = 256
max_iters = 2000
eval_interval = 500
eval_iters = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.1

torch.manual_seed(1337)

enc = RegexTokenizer()
enc.load("sen_twitter2.model")
print(enc.decode(enc.encode("السلام عليكم كيف الحال How Are You !")))

vocab_size = len(enc.vocab)

def load_process_dataset(filepath):
    # data = pd.read_csv(filepath).to_numpy()
    # labels = torch.tensor(data[:, 0].astype(int), dtype=torch.int32)

    # # Tokenize texts and store their lengths
    # tokenized = [enc.encode(str(text)) for text in data[:, 1]]  # Ensure text is string
    # ids = torch.tensor([[0] * (block_size - len(tokens)) + tokens[-block_size:] for tokens in tokenized])

    # torch.save((ids, labels), f"preprocessed_trainset{block_size}-2000-newds.pt")
    # exit()
    # Load existing pt
    ids, labels = torch.load("preprocessed_trainset512-2000-newds.pt", weights_only=False)
    ids = ids[:, -block_size:]
    # exit()
    return torch.concat([ids, labels.view(-1, 1)], dim=-1)


data = load_process_dataset("combined_dataset.csv")
num_classes = len(torch.unique(data[:, -1]))
print(f"Number of classes: {num_classes}")
print(data.shape) # last element is the label
# Shuffle
shuffle_idx = torch.randperm(data.shape[0])
data = data[shuffle_idx]

n = int(len(data) * 0.9)

train_data = data[:n]
val_data = data[n:]

print(train_data.shape, val_data.shape)
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - batch_size, (batch_size, ))
    x = data[ix, :-1]
    y = data[ix, -1:].squeeze()
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(pbar):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        k = self.key(x) # B, T, H
        q = self.query(x) # B, T, H
        v = self.value(x) # B, T, H

        # wei = q @ k.transpose(-2, -1) * C**-0.5 # B, T, T
        # if mask is not None:
        #     wei = wei.masked_fill(mask.unsqueeze(1).expand(B, T, T) == 0, float('-inf')) # B, T, T
        # wei = F.softmax(wei, dim=-1)
        # wei = self.dropout(wei)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask = mask.unsqueeze(1).expand(B, T, T), dropout_p=dropout) # B, T, H
        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out = torch.cat([h(x, mask=mask) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x, mask=None):
        x = self.ln1(x)
        x = x + self.sa(x, mask=mask)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x

class SentimentTransformerModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embbeding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, num_classes) 

    def forward(self, idx, targets=None):
        B, T = idx.shape

        mask = idx[:, -block_size:] != 0
        tok_emb = self.token_embbeding_table(idx[:, -block_size:])
        pos_emb = self.position_embedding_table(torch.arange(block_size, device=device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.ln_f(x)
        # Last token
        logits = self.lm_head(x[:, -1, :])

        if targets is None:
            loss = None
        else :
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx):
        idx_cond = idx[:, -block_size:]
        logits, loss = self(idx_cond)
        logits = logits
        probs = F.softmax(logits, dim=-1)
        print(probs)
        return torch.argmax(probs, dim=-1)


model = SentimentTransformerModel()
m = model.to(device)
# model.load_state_dict(torch.load("sentiment_analysis_v1-cs512-nl4.pth", weights_only=True))

num_parameters = sum(p.nelement() for p in model.parameters())
print(f"Parameters: {num_parameters}")

torch.set_float32_matmul_precision('high')

def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    running_loss = 0.0
    running_accuracy = 0.0
    with tqdm(total=max_iters, desc="Training", unit='iter') as pbar:
        for iter in range(max_iters):

            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss(pbar)
                tqdm.write(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                # model_save_path = "sentiment_analysis_v1"
                # torch.save(model.state_dict(), f"{model_save_path}-{iter}-cs{block_size}-nl{n_layer}.pth")
                # print(f"Model {model_save_path} Saved Successfully")


            X, Y = get_batch("train")

            logits, loss = model(X, Y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == Y).float().sum()
            accuracy = correct / batch_size
            running_accuracy = (0.99 * running_accuracy) + (accuracy * 0.01) if iter > 0 else accuracy
            running_loss = (0.99 * running_loss) + (loss * 0.01) if iter > 0 else loss
            pbar.set_description(f"Loss: {running_loss:.4f} | Acc: {running_accuracy:.4f} | Norm: {norm:.4f}")
            pbar.update(1)

    model_save_path = "sentiment_analysis_v1"
    torch.save(model.state_dict(), f"{model_save_path}-cs{block_size}-nl{n_layer}.pth")
    print(f"Model {model_save_path} Saved Successfully")

train()

def test():
    model.load_state_dict(torch.load("sentiment_analysis_v1-cs256-nl6.pth", weights_only=True))
    model.eval()  # Set model to evaluation mode
    
    print("\nSentiment Analysis Ready! Enter a sentence (E to exit):\n")

    while True:
        text = input("Text: ")
        if text.lower() == "e":
            print("Exiting sentiment analysis.")
            break

        # Tokenize input text
        encoded = enc.encode(text)
        tokenized = torch.tensor([ [0] * (block_size - len(encoded)) + encoded[-block_size:] ], dtype=torch.int32).to(device)

        # Get sentiment prediction
        with torch.no_grad():
            sentiment = model.generate(tokenized).item()

        # Print result
        print("Sentiment:", "😊 Positive" if sentiment == 1 else "☹️ Negative")

# test()
# The issue we have is 0 token's embeddings is getting involved in the shit
# we need to cut them out, the way we are doing the cut in line # 97 is not correct, there is nothing getting cut