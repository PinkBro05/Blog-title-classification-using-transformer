import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
import math

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.titles = dataframe['text'].str.lower().values
        self.labels = dataframe['labels'].values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = self.titles[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode(title)
        input_ids = torch.tensor(encoding, dtype=torch.long)
        return input_ids, label

# Collate function to pad sequences
def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    max_length = max(len(ids) for ids in input_ids)
    input_ids = torch.stack([torch.cat([ids, torch.zeros(max_length - len(ids), dtype=torch.long)]) for ids in input_ids])
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, labels

# Sigmoid Attention with multi-head support based on the correct formula
class SigmoidAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SigmoidAttention, self).__init__()
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # Learnable weight matrices for query, key, value
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, depth) and transpose
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)  # (batch_size, num_heads, seq_len, depth)

    def combine_heads(self, x, batch_size):
        # Transpose back and combine the heads
        x = x.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, depth)
        return x.view(batch_size, -1, self.num_heads * self.depth)  # (batch_size, seq_len, d_model)

    def forward(self, X):
        batch_size = X.size(0)
        seq_len = X.size(1)

        # Set b to -log(n), where n is the sequence length
        b = -torch.log(torch.tensor(seq_len, dtype=torch.float32))

        # Linear projections
        Q = self.W_q(X)  # (batch_size, seq_len, d_model)
        K = self.W_k(X)  # (batch_size, seq_len, d_model)
        V = self.W_v(X)  # (batch_size, seq_len, d_model)

        # Split heads for multi-head attention
        Q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len, depth)
        K = self.split_heads(K, batch_size)  # (batch_size, num_heads, seq_len, depth)
        V = self.split_heads(V, batch_size)  # (batch_size, num_heads, seq_len, depth)

        # Compute QK^T / sqrt(d_qk)
        dk = K.size(-1)  # depth
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))  # (batch_size, num_heads, seq_len, seq_len)

        # Apply the sigmoid function with b = -log(n)
        sigmoid_scores = torch.sigmoid(scores + b)

        # Compute the weighted sum of the values
        output = torch.matmul(sigmoid_scores, V)  # (batch_size, num_heads, seq_len, depth)

        # Combine heads back
        output = self.combine_heads(output, batch_size)  # (batch_size, seq_len, d_model)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# Transformer Encoder Layer with Sigmoid Attention
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        # Linear layers for query, key, value, and output
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization and dropout
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Sigmoid Attention
        self.sigmoid_attention = SigmoidAttention(d_model, num_heads)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # Use Sigmoid Attention
        concat_attention = self.sigmoid_attention(x)

        # Apply the final linear layer to combine the heads
        attn_output = self.dense(concat_attention)

        # Add & Norm
        x = self.layernorm1(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)

        # Add & Norm
        x = self.layernorm2(x + self.dropout(ff_output))

        return x

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, d_model, num_heads, d_ff, output_size, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x)  # (batch_size, seq_len, embed_size)
        x = self.positional_encoding(x)  # (batch_size, seq_len, d_model)
        for layer in self.encoder_layers:
            x = layer(x, mask)  # (batch_size, seq_len, d_model)
        x = x.mean(dim=1)  # (batch_size, d_model)
        x = self.fc(self.dropout(x))  # (batch_size, output_size)
        return x

def train_and_evaluate_model(params):
    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    dataset_df = pd.read_csv('data/formatted.csv')
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    
    dataset = TextDataset(dataset_df, tokenizer)
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.1, shuffle=True)

    batch_size = params['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Initialize the Transformer model
    vocab_size = tokenizer.vocab_size
    d_model = params['d_model']  # Ensure embed_size matches d_model
    embed_size = d_model
    num_heads = params['num_heads']
    d_ff = params['d_ff']
    output_size = len(dataset_df['labels'].unique())
    num_layers = params['num_layers']
    dropout = params['dropout']

    model = TransformerModel(vocab_size, embed_size, d_model, num_heads, d_ff, output_size, num_layers, dropout).to(device)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='runs/experiment1')

    # Training loop
    num_epochs = params['num_epochs']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=0.01)

    # Define the learning rate schedule with warm-up
    def lr_lambda(current_step):
        warmup_steps = 0.1 * num_epochs * len(train_loader)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(num_epochs * len(train_loader) - current_step) / float(max(1, num_epochs * len(train_loader) - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, (input_ids, labels) in enumerate(train_loader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            if i % 10 == 9:  # Log every 10 batches
                writer.add_scalar('training_loss', running_loss / 10, epoch * len(train_loader) + i)
                running_loss = 0.0

        train_accuracy = 100 * correct_train / total_train
        writer.add_scalar('training_accuracy', train_accuracy, epoch)

        # Validation step
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        writer.add_scalar('validation_loss', val_loss / len(val_loader), epoch)
        writer.add_scalar('validation_accuracy', val_accuracy, epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        # Saving checkpoint for each epoch
        torch.save({'model': model.state_dict(), 'loss': loss.item()}, f"checkpoint/8/model_checkpoint_epoch_{epoch+1}.pt")

    writer.close()
    return model

def main():
    # # Define the hyperparameter grid
    # param_grid = {
    #     'batch_size': [16, 32],
    #     'd_model': [128, 256],  # Ensure embed_size matches d_model
    #     'num_heads': [4, 8],
    #     'd_ff': [256, 512],
    #     'num_layers': [2, 4],
    #     'dropout': [0.1, 0.3],
    #     'learning_rate': [0.001, 0.0001],
    #     'num_epochs': [10, 20]
    # }

    # # Perform grid search
    # best_accuracy = 0
    # best_params = None
    # for params in ParameterGrid(param_grid):
    #     print(f"Training with params: {params}")
    #     accuracy = train_and_evaluate_model(params)
    #     if accuracy > best_accuracy:
    #         best_accuracy = accuracy
    #         best_params = params

    # print(f"Best Accuracy: {best_accuracy:.2f}%")
    # print(f"Best Hyperparameters: {best_params}")
    
    best_params = {
        'batch_size': 32,
        'd_model': 384,
        'num_heads': 8,
        'd_ff': 1024,
        'num_layers': 5,
        'dropout': 0.3,
        'learning_rate': 3e-5,
        'num_epochs': 20
    }
    model = train_and_evaluate_model(best_params)

if __name__ == "__main__":
    main()