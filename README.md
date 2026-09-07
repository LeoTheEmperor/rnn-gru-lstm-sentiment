# Comparative Analysis of RNN, GRU & LSTM for Sentiment Classification

A comparative study of three recurrent architectures — Vanilla RNN, GRU,
and LSTM — for binary sentiment classification on a toxic-text dataset.

## Problem
The dataset is heavily imbalanced (~29,720 non-toxic vs. ~2,242 toxic
tweets, where toxic includes sexist/racist content). Rather than
optimizing directly on this skew, the dataset was balanced via
undersampling to 4,484 samples, then used to fairly compare how RNN, GRU,
and LSTM handle the same classification task under identical conditions.

## Approach
- Balanced the dataset via undersampling the majority (non-toxic) class
- Cleaned text: lowercased, stripped URLs/mentions, normalized whitespace
- Tokenized and padded sequences to a fixed length (42 tokens) using the
  Keras Tokenizer
- Trained a 128-dimensional embedding layer from scratch (no pretrained
  embeddings) for the core comparison
- All three models share an identical downstream architecture (128-unit
  recurrent layer → Dense(256) → BatchNorm → Dense(128) → BatchNorm →
  sigmoid output) so any performance difference comes from the recurrent
  layer choice alone
- Trained under identical conditions: Adam optimizer (lr=0.001), binary
  crossentropy, 10 epochs, batch size 64, 10% validation split
- Evaluated on an 80–20 stratified train/test split using accuracy,
  precision, recall, and F1

## Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| RNN   | 0.82     | 0.83      | 0.82   | 0.82 |
| GRU   | 0.84     | 0.84      | 0.84   | 0.84 |
| LSTM  | 0.85     | 0.86      | 0.85   | 0.85 |

LSTM achieved the highest overall accuracy (85%) and notably the best
recall on the toxic class (90%), making it the strongest choice for
catching harmful content. GRU offered a close, more computationally
efficient alternative with more stable validation behavior and less
overfitting than the Vanilla RNN. All models showed some degree of
overfitting on this moderate-sized dataset, most noticeably the Vanilla
RNN.

## Extension: Word2Vec embeddings
The notebook also includes a fourth variant using a custom-trained
Word2Vec (skip-gram) embedding feeding into an LSTM, as an exploratory
extension beyond the core three-model comparison. Results for this
variant are experimental and not included in the table above.

## Tech stack
Python, TensorFlow/Keras, scikit-learn, pandas, matplotlib, Gensim
(Word2Vec extension)

## Dataset
[Toxic Tweet Dataset (Kaggle)](https://www.kaggle.com/datasets/vikasukani/hate-speech-and-offensive-content-identification)
