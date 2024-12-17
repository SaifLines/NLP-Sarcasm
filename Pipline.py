import argparse
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Download NLTK resources
nltk.download('stopwords')

# Load contractions
contractions = pd.read_csv('contractions.csv', index_col='Contraction', encoding='latin-1')
contractions.index = contractions.index.str.lower()
contractions.Meaning = contractions.Meaning.str.lower()
contractions_dict = contractions.to_dict()['Meaning']

# Regex patterns
urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"  # URLs
userPattern = '@[^\s]+'                                        # @USERNAME
hashtagPattern = r'#\S+'                                       # Hashtags
sequencePattern = r"(.)\1\1+"                                  # Consecutive letters
seqReplacePattern = r"\1\1"                                    # Replace long sequences
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"    # General text cleaning

# NLTK components
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

# Preprocessing function: Replace contractions, remove patterns, normalize text
def preprocess_apply(tweet):
    tweet = tweet.lower()
    # Remove URLs
    tweet = re.sub(urlPattern, '', tweet)
    # Remove @USERNAME mentions
    tweet = re.sub(userPattern, '', tweet)
    # Remove hashtags
    tweet = re.sub(hashtagPattern, '', tweet)
    # Replace 3+ consecutive letters by 2 letters
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
    # Replace contractions
    for contraction, replacement in contractions_dict.items():
        tweet = tweet.replace(contraction, replacement)
    # Add spaces around slashes
    tweet = re.sub(r'/', ' / ', tweet)
    return tweet.strip()

# Preprocessing function: Remove stop words, apply stemming
def preprocess(tweet, stem=True):
    tweet = re.sub(text_cleaning_re, ' ', str(tweet).lower()).strip()
    tokens = []
    for token in tweet.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

# Encode sentences using tokenizer
def encode_sentences(sentences, tokenizer, max_length):
    encoded_ids = []
    for sentence in sentences:
        encoding = tokenizer.encode_plus(
            sentence,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=False
        )
        encoded_ids.append(encoding['input_ids'])
    return tf.convert_to_tensor(encoded_ids)

# Build model architecture
def build_model(pretrained_model_name, max_length, dense_units, dropout_rate, learning_rate):
    base_model = TFAutoModel.from_pretrained(pretrained_model_name)
    input_word_ids = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="input_word_ids")
    embedding = base_model(input_word_ids).last_hidden_state[:, 0, :]  # Extract [CLS] token
    dense = tf.keras.layers.Dense(dense_units, activation='relu')(embedding)
    dense = tf.keras.layers.Dropout(dropout_rate)(dense)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
    model = tf.keras.Model(inputs=input_word_ids, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Main function
def main(args):
    # Load dataset
    print("Loading dataset...")
    data = pd.read_json(args.dataset, lines=True) if args.dataset.endswith(".json") else pd.read_csv(args.dataset)
    
    # Preprocess text
    print("Preprocessing text...")
    data['processed_text'] = data[args.text_column].apply(preprocess_apply)
    data['processed_text'] = data['processed_text'].apply(preprocess)

    labels = data[args.label_column].values
    sentences = data['processed_text'].values

    # Split data
    train_sents, test_sents, train_labels, test_labels = train_test_split(sentences, labels, test_size=0.15)

    # Load tokenizer
    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # Encode sentences
    print("Encoding sentences...")
    train_ids = encode_sentences(train_sents, tokenizer, args.max_length)
    test_ids = encode_sentences(test_sents, tokenizer, args.max_length)

    train_labels = tf.convert_to_tensor(train_labels)
    test_labels = tf.convert_to_tensor(test_labels)

    # Build and train the model
    print("Building model...")
    model = build_model(
        pretrained_model_name=args.model,
        max_length=args.max_length,
        dense_units=args.dense_units,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate
    )

    print("Training model...")
    history = model.fit(
        x=train_ids,
        y=train_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(test_ids, test_labels),
        verbose=1
    )

    # Evaluate the model
    print("Evaluating model...")
    predictions = model.predict(test_ids)
    predictions = (predictions.flatten() > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(test_labels.numpy(), predictions, target_names=["Not Sarcastic", "Sarcastic"]))

# Argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary Text Classification with Pretrained Transformers and Preprocessing")

    # Dataset and column arguments
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file (CSV or JSON)")
    parser.add_argument("--text_column", type=str, default="headline", help="Name of the text column")
    parser.add_argument("--label_column", type=str, default="is_sarcastic", help="Name of the label column")

    # Model and tokenizer arguments
    parser.add_argument("--model", type=str, required=True, help="Name of the pretrained model (e.g., 'bert-base-uncased')")
    parser.add_argument("--max_length", type=int, default=16, help="Maximum token length for input sentences")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")
    parser.add_argument("--dense_units", type=int, default=128, help="Number of units in the dense layer")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")

    args = parser.parse_args()
    main(args)
# command  python Pipline.py 
# --dataset sarcasm_dataset.json 
# --text_column headline 
# --label_column is_sarcastic 
# --model bert-base-uncased 
# --epochs 3 --
# batch_size 32 --
# max_length 16 
# --learning_rate 1e-5