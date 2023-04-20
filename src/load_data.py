import os
import requests
import re
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
import torch

def download_data(path="./data", filename="/shakespeare.txt"):
    # Download data if not already downloaded
    if not os.path.isfile(path+filename):
        if not os.path.exists(path):
            os.makedirs(path)
        r = requests.get("https://www.gutenberg.org/files/100/100-0.txt", stream = True)
        r.encoding = "utf-8-sig"
        with open(path+filename, "w", encoding='UTF-8') as f:
            f.write(r.text)

def load_data(path='data/shakespeare.txt') -> list[str]:
    with open(path, "r", encoding='UTF-8') as f:
        data = f.read()

    # Remove newlines, brackets, and apostrophes
    data = data.replace("\n", " ")
    data = re.sub("[’'\[\]\(\)_]", "", data)
    data = re.sub(" +", " ", data)
    words = re.split(r"\b", data)

    # Remove spaces and capitalization
    to_remove = [" ", ""]
    words = [word.lower().strip() for word in words if word not in to_remove]
    
    print(f"Data size: {len(words)} words")
    return words

def create_word_dicts(words: list[str], min_occurrences=10) -> tuple[dict[str, int], dict[int, str]]:
    # Filter for words with <n occurrences
    word_counts = Counter(words)
    high_occurence_words: list[str] = []
    low_occurence_words: list[str] = []
    for key, value in word_counts.items():
        if value >= min_occurrences:
            high_occurence_words.append(key)
        else:
            low_occurence_words.append(key)

    
    # Create embedding
    word_to_index = {word:i for i, word in enumerate(high_occurence_words)}
    index_to_word = {i:word for i, word in enumerate(high_occurence_words)}
    last_index = len(word_to_index)
    for word in low_occurence_words:
        word_to_index[word] = last_index
        index_to_word[last_index] = "<unknown>"

    print(f"{len(high_occurence_words)} words that occur >= {min_occurrences} times")
    print(f"{len(low_occurence_words)} words that occur < {min_occurrences} times")
    print(f"Vocabulary size: {len(index_to_word)}")

    return word_to_index, index_to_word

def create_dataset(words, word_to_index, index_to_word, batch_size=64, val_split=0.2, max_input_length=30):
    # Tokenize dataset
    tokenized_data = [word_to_index[word] for word in words]
    labels = tokenized_data[1:] + [len(index_to_word) - 1]

    # Create pytorch dataset
    cutoff = len(tokenized_data) % max_input_length
    data_x = tokenized_data[:-cutoff]
    data_y = labels[:-cutoff]
    data_x = torch.LongTensor(data_x).view(-1, max_input_length)
    data_y = torch.LongTensor(data_y).view(-1, max_input_length)
    ds = TensorDataset(data_x, data_y)
    val_size = int(len(ds)*val_split)
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds)-val_size, val_size])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl