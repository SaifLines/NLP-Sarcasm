# Text Classification Experiment Framework

This repository provides a flexible script to run text classification experiments using various datasets, pretrained models, and configurations. You can easily customize and execute experiments directly from the terminal.

## Features
- Supports multiple datasets in JSON format.
- Utilizes pretrained models from Hugging Face Transformers.
- Option to include additional textual features in the model.
- Flexible parameter customization (e.g., batch size, learning rate, etc.).
- Generates classification reports and training metrics.

## Prerequisites

1. **Python 3.7+**
2. Required libraries:
   - `numpy`
   - `pandas`
   - `tensorflow`
   - `transformers`
   - `sklearn`
   - `nltk`

Install the required libraries using:
```bash
pip install -r requirements.txt
```

## Usage

Run the script with customizable parameters:

```bash
python script_name.py \
--dataset <path_to_dataset> \
--model <pretrained_model_name> \
[--use_features] \
--batch_size <batch_size> \
--learning_rate <learning_rate> \
--dense_units <dense_units> \
--dropout_rate <dropout_rate> \
--epochs <num_epochs> \
--max_length <max_length> \
--test_size <test_size>
```

### Example Command

```bash
python script_name.py \
--dataset "./datasets/sarcasm.json" \
--model "roberta-base" \
--use_features \
--batch_size 16 \
--learning_rate 2e-5 \
--dense_units 256 \
--dropout_rate 0.3 \
--epochs 5 \
--max_length 64 \
--test_size 0.2
```

### Parameters
- `--dataset`: Path to the dataset JSON file.
- `--model`: Pretrained model name (e.g., `roberta-base`, `xlnet-base-cased`, etc.).
- `--use_features`: (Optional) Flag to include additional features in the model.
- `--batch_size`: Batch size for training (default: 32).
- `--learning_rate`: Learning rate for the optimizer (default: 1e-5).
- `--dense_units`: Number of units in the dense layer (default: 128).
- `--dropout_rate`: Dropout rate for the dense layer (default: 0.2).
- `--epochs`: Number of epochs for training (default: 3).
- `--max_length`: Maximum sequence length for tokenization (default: 32).
- `--test_size`: Proportion of data to use for testing (default: 0.15).

## Dataset Format
The dataset should be in JSON format, with at least the following fields:
- `headline`: Text data to classify.
- `is_sarcastic`: Binary label for classification.

Example:
```json
{
    "headline": "This is a sarcastic example.",
    "is_sarcastic": 1
}
```

## Output
- Classification report with precision, recall, and F1-score.
- Training metrics (accuracy and loss) plotted for visualization.


