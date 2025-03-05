import torch
import pandas as pd
from encoder import TransformerModel
from transformers import AutoTokenizer
from sklearn.metrics import precision_score

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

# Initialize the Transformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

vocab_size = tokenizer.vocab_size
d_model = best_params['d_model']  # Ensure embed_size matches d_model
num_heads = best_params['num_heads']
d_ff = best_params['d_ff']
output_size = 9
num_layers = best_params['num_layers']
dropout = best_params['dropout']

model = TransformerModel(vocab_size, d_model, num_heads, d_ff, output_size, num_layers, dropout).to(device)
checkpoint = torch.load("checkpoint/4/model_checkpoint_epoch_19.pt")
model.load_state_dict(checkpoint['model'])

def predict_sentence(model, tokenizer, sentence):
    """
    Dự đoán nhãn cho một câu sử dụng mô hình GRU đã huấn luyện.

    Parameters:
    - model: Mô hình GRU đã huấn luyện.
    - tokenizer: GPT tokenizer đã được khởi tạo.
    - sentence: Câu cần dự đoán (chuỗi văn bản).

    Returns:
    - predicted_label: Nhãn dự đoán cho câu (số nguyên).
    """
    model.eval()
    sentence = sentence.lower()
    input_ids = tokenizer.encode(sentence)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output = model(input_ids)
    _, predicted_label = torch.max(output, dim=1)

    return predicted_label.item()

# Đọc tập test từ file CSV
test_df = pd.read_csv('data/test_set_public.csv')

# Dự đoán nhãn cho từng tiêu đề trong tập test
test_df['label_numeric'] = test_df['title'].apply(lambda x: predict_sentence(model, tokenizer, x))

test_df.rename(columns={'_id': 'id'}, inplace=True)

test_df = test_df.drop('title', axis=1)

# Lưu kết quả vào file CSV
test_df.to_csv('result/4/your_submissions.csv', index=False)
