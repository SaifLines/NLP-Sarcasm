import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import nltk
from transformers import AutoTokenizer, TFAutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import os
import argparse

nltk.download('stopwords')
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

# Define text preprocessing functions
def preprocess_apply(tweet, contractions_dict):
    tweet = tweet.lower()
    tweet = re.sub(r"http[s]?://\S+", "", tweet)
    tweet = re.sub(r"@\S+", "", tweet)
    tweet = re.sub(r"[^A-Za-z0-9]+", " ", tweet)
    for contraction, replacement in contractions_dict.items():
        tweet = tweet.replace(contraction, replacement)
    return tweet.strip()

def preprocess(tweet, stem=True):
    tokens = [stemmer.stem(word) if stem else word for word in tweet.split() if word not in stop_words]
    return " ".join(tokens)

# Feature extraction function
def extract_features(text):
    return [
        len(re.findall(r'[\U0001F600-\U0001F64F]', text)),
        len(re.findall(r'[!?.]', text)),
        len(re.findall(r'#\S+', text)),
        len(re.findall(r'@\S+', text)),
        sum(1 for c in text if c.isupper()),
        len(re.findall(r'(.)\1{2,}', text)),
        len(re.findall(r'\b(very|so|totally|really)\b', text, re.IGNORECASE)),
        len(re.findall(r'\b(oh|wow|oops|ugh|eh)\b', text, re.IGNORECASE))
    ]

# Encode sentences and features
def encode_inputs(sentences, tokenizer, max_length, use_features=False):
    input_ids = []
    extra_features = []

    for sentence in sentences:
        encoding = tokenizer.encode_plus(
            sentence,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=False
        )
        input_ids.append(encoding['input_ids'])

        if use_features:
            features = extract_features(sentence)
            extra_features.append(features)

    return np.array(input_ids), (np.array(extra_features) if use_features else None)

# Build the model
def build_model(pretrained_model_name, feature_size, max_length, dense_units, dropout_rate, learning_rate, use_features=False):
    text_input = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="text_input")
    transformer_model = TFAutoModel.from_pretrained(pretrained_model_name)
    transformer_output = transformer_model(text_input).last_hidden_state[:, 0, :]

    if use_features:
        extra_input = tf.keras.Input(shape=(feature_size,), dtype=tf.float32, name="extra_features")
        combined = tf.keras.layers.Concatenate()([transformer_output, extra_input])
    else:
        combined = transformer_output

    dense = tf.keras.layers.Dense(dense_units, activation='relu')(combined)
    dense = tf.keras.layers.Dropout(dropout_rate)(dense)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

    inputs = [text_input] if not use_features else [text_input, extra_input]
    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def main(args):
    print(f"Using Dataset: {args.dataset}")
    print(f"Using Pretrained Model: {args.model}")

    # Load dataset
    data = pd.read_json(args.dataset, lines=True)
    contractions_dict = {"can't": "cannot", "n't": "not"}  # Add more contractions as needed

    if args.use_features:
        data['features'] = data['headline'].apply(lambda x: extract_features(x))

    data['processed_text'] = data['headline'].apply(lambda x: preprocess_apply(x, contractions_dict)).apply(preprocess)

    labels = data['is_sarcastic'].values
    sentences = data['processed_text'].values

    train_sents, test_sents, train_labels, test_labels = train_test_split(
        sentences, labels, test_size=args.test_size, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    max_length = args.max_length

    train_text_ids, train_features = encode_inputs(train_sents, tokenizer, max_length, args.use_features)
    test_text_ids, test_features = encode_inputs(test_sents, tokenizer, max_length, args.use_features)

    feature_size = train_features.shape[1] if args.use_features else 0

    model = build_model(
        args.model, feature_size, max_length,
        dense_units=args.dense_units,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        use_features=args.use_features
    )

    print("Training model...")
    if args.use_features:
        model.fit(
            [train_text_ids, train_features], train_labels,
            validation_data=([test_text_ids, test_features], test_labels),
            epochs=args.epochs, batch_size=args.batch_size
        )
    else:
        model.fit(
            train_text_ids, train_labels,
            validation_data=(test_text_ids, test_labels),
            epochs=args.epochs, batch_size=args.batch_size
        )

    predictions = model.predict([test_text_ids, test_features] if args.use_features else test_text_ids)
    predictions = (predictions.flatten() > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=["Not Sarcastic", "Sarcastic"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text classification experiments.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset JSON file.")
    parser.add_argument("--model", type=str, required=True, help="Pretrained model name (e.g., roberta-base).")
    parser.add_argument("--use_features", action='store_true', help="Whether to use additional features.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--dense_units", type=int, default=128, help="Number of units in the dense layer.")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate for the dense layer.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--max_length", type=int, default=32, help="Maximum sequence length for the tokenizer.")
    parser.add_argument("--test_size", type=float, default=0.15, help="Proportion of data to use for testing.")

    args = parser.parse_args()
    main(args)
#python Pipline.py 
# --dataset Ghosh/sarcasm_dataset.json 
# --model roberta-base 
# --use_features 
# --batch_size 32 
# --learning_rate 1e-5 
# --dense_units 256  
# --dropout_rate 0.2 
# --epochs 1 
# --max_length 32 
# --test_size 0.2  