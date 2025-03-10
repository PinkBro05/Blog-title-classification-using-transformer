# import pandas as pd
# import re
# import string
# import matplotlib.pyplot as plt

# # Function to normalize Vietnamese text
# def normalize_vietnamese_text(text):
#     text = text.lower()  # 1. Convert to lowercase
#     text = text.replace('"', '')  # 2. Remove the character "
#     text = re.sub(r'\s+', ' ', text).strip()  # 3. Remove extra spaces
#     return text

# # Load the formatted.csv file
# df = pd.read_csv('data/formatted.csv')

# # Apply the normalization function to the 'text' column
# df['text'] = df['text'].apply(normalize_vietnamese_text)

# # # Calculate the number of words in each text
# # df['num_words'] = df['text'].apply(lambda x: len(x.split()))

# # # Get the minimum and maximum number of words
# # min_words = df['num_words'].min()
# # max_words = df['num_words'].max()

# # print(f"Minimum number of words: {min_words}")
# # print(f"Maximum number of words: {max_words}")

# # # Plot the distribution of the number of words
# # plt.figure(figsize=(10, 6))
# # df['num_words'].hist(bins=30)
# # plt.xlabel('Number of Words')
# # plt.ylabel('Frequency')
# # plt.title('Distribution of Number of Words in Texts')
# # plt.show()

# # Save the processed data to formatted_processed.csv
# df.to_csv('data/formatted_processed.csv', index=False)

import random
import pandas as pd
from deep_translator import GoogleTranslator

# Load Dataset
df = pd.read_csv("data/formatted.csv")

# Synonym Dictionary (Extend as needed)
synonym_dict = {
    "tuyệt vời": ["xuất sắc", "hoàn hảo", "ấn tượng"],
    "rẻ": ["giá tốt", "hợp lý", "tiết kiệm"],
    "đánh giá": ["nhận xét", "phản hồi", "review"],
    "tốt": ["tuyệt", "ổn", "đáng tiền"],
    "mua": ["sắm", "tậu", "đặt hàng"]
}

def replace_synonyms(text):
    words = text.split()
    new_words = [random.choice(synonym_dict[word]) if word in synonym_dict else word for word in words]
    return " ".join(new_words)

def back_translate(text):
    try:
        temp = GoogleTranslator(source="vi", target="en").translate(text)
        translated_back = GoogleTranslator(source="en", target="vi").translate(temp)
        return translated_back
    except:
        return text  # Return original text if translation fails

def random_deletion(text, p=0.1):
    words = text.split()
    if len(words) == 1: return text  # Avoid deleting all words
    new_words = [word for word in words if random.uniform(0, 1) > p]
    return " ".join(new_words) if new_words else words[0]

def random_insertion(text, n=1):
    words = text.split()
    for _ in range(n):
        insert_pos = random.randint(0, len(words))
        random_word = random.choice(list(synonym_dict.keys()))  # Insert a random synonym
        words.insert(insert_pos, random_word)
    return " ".join(words)

# Apply augmentation to minority class samples
augmented_data = []
class_counts = df['labels'].value_counts()
min_class = class_counts.idxmin()  # Find the minority class

for _, row in df.iterrows():
    text, label = row['text'], row['labels']
    augmented_data.append((text, label))  # Original
    if label == min_class:
        augmented_data.append((replace_synonyms(text), label))  # Synonym Replacement
        augmented_data.append((back_translate(text), label))  # Back Translation
        augmented_data.append((random_deletion(text), label))  # Word Deletion
        augmented_data.append((random_insertion(text), label))  # Word Insertion

# Convert to DataFrame and Save
df_augmented = pd.DataFrame(augmented_data, columns=['text', 'labels'])
df_augmented.to_csv("data/augmented_dataset.csv", index=False)

print("✅ Augmentation complete! Dataset saved to data/augmented_dataset.csv")
