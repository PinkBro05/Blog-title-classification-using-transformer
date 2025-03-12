# Blog-title-classification-using-transformer
Build a Classification using Transformer to classify Vietnamese blog title

Framework: PyTorch, transformer

Architecture: Encoder only 

Problem: short text classification

Dataset insight: 4k3 rows, major of word is capital, extra space. Very imbalanced, 1 label dominate 70% and some only <1%.

Base architecture result: 74% accuracy - Sinusoid/GPT2 tokenizer/Multiheaded attention/masking/{
    #     'batch_size': [32],
    #     'd_model': [256],
    #     'num_heads': [4],
    #     'd_ff': [256],
    #     'num_layers': [4],
    #     'dropout': [0.1],
    #     'learning_rate': [0.001],
    #     'num_epochs': [10]
}
Adam optimizer

Best architecture result: 81.7% accuracy - Sinusoid/PhoBert tokenizer/Multiheaded attention/masking/{
    #     'batch_size': [32],
    #     'd_model': [256],
    #     'num_heads': [4],
    #     'd_ff': [256],
    #     'num_layers': [4],
    #     'dropout': [0.1],
    #     'learning_rate': [0.001],
    #     'num_epochs': [10]
}
AdamW optimizer/Cosine Schduler/Gradient Clipping/ Data preprocessing

Technique explored in this project:
- ROPE (Rotary Positional Embedding): It requires more computational resource making the time for each epoch longer , idea is to use ROPE to keep semantic of data when positional embedding but it doesn't help in this case.
- Sigmoid Attention: instead of Softmax, Sigmoid attention is useful when there is short text and multi-label. Although it's not fit with this case but it doesn't reduce much accuracy but longer training time.
- Hierarchical Attention Network: try to focus more on words and sentence level since Vietnamese is different structure with English. The HAN architecture doesn't improve model much but its worth to implement it with the best result' profile.
- Label Smoothing: This technique improves accuracy of model but only with model below 80% of validation accuracy. Hypothesis that if it above 80% accuracy, the predict ability of model is already good on major class but minor and label smoothing may make the predict messup and reduce accuracy.
- Data preprocessing: Using dataset lower case and remove extra space. Help the model focus on the content not the capital.
- Data Augmentation: Double translate data(en-vn,vn-en), using Synonym, random delete and insert. This doesn't help improve accuracy much but maybe ChatGPT is not fit with generate synthetic data.
- Focal Loss: Assign weight to label so that it can deal with imbalanced data.
- Hyper-parameter tuning: Early stopping, tuning model's parameter, ReduceLROnPLateau 

Things can explore more:
- Pre-train the model with Vietnamese data
