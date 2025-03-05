import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tiktoken
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

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        # Linear layers for Q, K, V matrices
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        # Output linear transformation
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

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = torch.tensor(k.size(-1), dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        if mask is not None:
            scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, -1e9)

        attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)

        return output, attention_weights

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        q = self.split_heads(self.wq(x), batch_size)
        k = self.split_heads(self.wk(x), batch_size)
        v = self.split_heads(self.wv(x), batch_size)

        scaled_attention, _ = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.transpose(1, 2).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        attn_output = self.dense(concat_attention)

        x = self.layernorm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)

        x = self.layernorm2(x + self.dropout(ff_output))

        return x

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

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, output_size, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  # Ensure embed_size matches d_model
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        x = x.mean(dim=1)
        x = self.fc(self.dropout(x))
        return x

def train_and_evaluate_model(params):
    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    dataset_df = pd.read_csv('data/formatted.csv')
    
    tokenizer = tiktoken.get_encoding('gpt2')
    
    dataset = TextDataset(dataset_df, tokenizer)
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.1)

    batch_size = params['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Initialize the Transformer model
    vocab_size = tokenizer.n_vocab
    d_model = params['d_model']  # Ensure embed_size matches d_model
    num_heads = params['num_heads']
    d_ff = params['d_ff']
    output_size = len(dataset_df['labels'].unique())
    num_layers = params['num_layers']
    dropout = params['dropout']

    model = TransformerModel(vocab_size, d_model, num_heads, d_ff, output_size, num_layers, dropout).to(device)

    # Training loop
    num_epochs = params['num_epochs']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    for epoch in range(num_epochs):
        model.train()
        for input_ids, labels in train_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                outputs = model(input_ids)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Validation Accuracy: {accuracy:.2f}%")
        
        #Saving checkpoint for each epoch
        torch.save({'model': model.state_dict(), 'loss': loss.item()}, f"checkpoint/1/model_checkpoint_epoch_{epoch}.pt")

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
        'd_model': 256,
        'num_heads': 8,
        'd_ff': 256,
        'num_layers': 2,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'num_epochs': 20
    }
    model = train_and_evaluate_model(best_params)


if __name__ == "__main__":
    main()