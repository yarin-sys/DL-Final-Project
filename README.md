# Tweet-Level and Topic-Based Sentiment Analysis on Twitter (SemEval-2017 Task 4)

This repository contains the code and report for our Deep Learning final project:

> **Tweet-Level and Topic-Based Sentiment Analysis on Twitter with BERTweet and RoBERTa: A SemEval-2017 Task 4 Study**

We fine-tune transformer-based models on the **SemEval-2017 Task 4** Twitter sentiment benchmark and evaluate them on:

- **Subtask A – Tweet-level sentiment classification (3-way):**  
  Predict the overall sentiment of a tweet as **positive**, **neutral**, or **negative**.
- **Subtask B – Topic-based (targeted) sentiment classification (2-way):**  
  Given a **(topic, tweet)** pair, predict the sentiment **towards the topic** as **positive** or **negative**.

The project compares **BERTweet-base** (Twitter-specific pretraining) and **RoBERTa-base** (general-domain pretraining), and uses a **cross-encoder architecture** for topic-based sentiment.

---

## 1. Project Structure

A suggested repository layout for this project is:

```text
.
├── deep_learning_subtaskA_bertweet_roberta.ipynb   # Notebook for Subtask A (tweet-level sentiment)
├── deep_learning_subtaskB_bertweet_roberta.ipynb   # Notebook for Subtask B (topic-based sentiment, cross-encoder)
├── 2017_English_final.zip # Final report #Dataset
└── README.md
```

> **Note:** The SemEval datasets are *not* included in this repo because of licensing. Please download them from the official sources (see **Datasets** section) and place them under the `data/` directory.

---

## 2. Tasks

### 2.1 Subtask A – Tweet-Level Sentiment (3-way)

- **Input:** A single tweet in English.
- **Output classes:** `positive`, `neutral`, `negative`.
- **Problem type:** Multi-class text classification.
- **Main metric:** **Macro-average recall** over the three classes.

We treat Subtask A as a standard single-sequence classification problem:

- The tweet is tokenized and fed into a transformer (BERTweet or RoBERTa).
- We use the **[CLS] (pooled)** output as the tweet representation.
- A linear classification head with softmax predicts the 3-way sentiment.

Because the dataset is **imbalanced** (negative and neutral are under-represented), we prioritize **macro-average recall** so that performance on minority classes still matters.

### 2.2 Subtask B – Topic-Based Sentiment (2-way)

- **Input:** A **(topic, tweet)** pair.
- **Output classes:** `positive`, `negative` (sentiment *towards the topic*).
- **Problem type:** Binary classification with multiple topics.
- **Main metric:** **Macro-average recall** over both classes, averaged over topics.

We use a **cross-encoder** architecture:

- The input sequence is:  
  `"[CLS] {topic} [SEP] {tweet} [SEP]"`.
- The model jointly encodes topic and tweet, allowing self-attention to capture their interaction.
- The pooled [CLS] representation is fed into a 2-way classification head.

---

## 3. Datasets

### 3.1 Subtask A – SemEval-2017 Task 4A (English Twitter Sentiment)

- Official shared task: **SemEval-2017 Task 4 – Sentiment Analysis in Twitter (Subtask A)**.
- Language: English.
- Labels: `positive`, `neutral`, `negative`.
- After deduplication by tweet ID, we keep ~49k unique tweets.
- Example split used in our experiments:
  - ~27k train
  - ~2k dev
  - ~20.5k test

To obtain the data:

1. Visit the official SemEval 2017 Task 4 website.
2. Request/download the English Twitter sentiment dataset.
3. Place the files into `data/subtaskA/`, adjusting paths inside the notebooks if needed.

### 3.2 Subtask B – SemEval 2015–2016 Topic-Based Sentiment

For topic-based sentiment, we combine English datasets from the **SemEval 2015–2016** targeted sentiment subtasks (e.g., Subtask B), then:

- Merge the official training and test files from multiple years.
- Filter **only** `positive` and `negative` labels (remove neutral / off-topic).
- Deduplicate by `(topic, tweet_id)`.

Resulting data (approximate):

- ~19k (topic, tweet) pairs after filtering.
- Split:
  - ~13k train
  - ~2.8k dev
  - ~2.8k test
- Positive is the majority class (~79%), negative is minority (~21%).

To obtain the data:

1. Download the SemEval 2015–2016 topic-based sentiment datasets from the official SemEval sites.
2. Place the files in `data/subtaskB/`.
3. Use the preprocessing cells in `deep_learning_subtaskB_bertweet_roberta.ipynb` to:
   - Merge the files,
   - Filter labels,
   - Build the train/dev/test splits.

---

## 4. Models

We use two transformer backbones via HuggingFace:

- **BERTweet-base**
  - RoBERTa-like architecture
  - Pre-trained on ~850M English tweets
  - Well-suited for noisy social media text (hashtags, mentions, emojis, etc.)

- **RoBERTa-base**
  - General-purpose transformer
  - Pre-trained on large English web corpora
  - Strong baseline for many text classification tasks

For both Subtask A and Subtask B we use:

- `AutoTokenizer` to tokenize the inputs.
- `AutoModelForSequenceClassification` to add a classification head on top of the encoder.

---

## 5. Training Setup

The notebooks are designed to be run in **Google Colab** (GPU recommended). We use similar training settings for all experiments:

- **Optimizer:** AdamW
- **Learning rate:** ~2e-5 (with linear warmup & decay)
- **Batch size:** 16
- **Epochs:** 3–4
- **Max sequence length:** 128 tokens
- **Gradient clipping:** max norm 1.0
- **Early stopping / model selection:** best **dev macro-average recall**

### 5.1 Running the Notebooks in Colab

1. Upload the repository (or individual notebooks and data) to your Google Drive.
2. Open:
   - `deep_learning_subtaskA_bertweet_roberta.ipynb` for Subtask A, and/or
   - `deep_learning_subtaskB_bertweet_roberta.ipynb` for Subtask B.
3. In the first cells, adjust:
   - `DATA_DIR` paths,
   - Any other file paths as needed.
4. Run all cells:
   - Preprocessing,
   - Model loading,
   - Training,
   - Evaluation.

### 5.2 Running Locally (Optional)

If you want to run everything locally instead of Colab:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install torch transformers scikit-learn pandas numpy tqdm

# 3. Launch Jupyter Lab / Notebook
pip install jupyterlab
jupyter lab
```

Then open the `.ipynb` files and run them.

> **Note:** Depending on your GPU and PyTorch version, you may need to adjust the `torch` installation command.

---

## 6. Evaluation & Metrics

We report:

- **Accuracy**  
- **Macro-average recall** (main metric for both subtasks)  
- **F1 scores**:
  - For Subtask A: F1 on positive & negative (`F1_PN`),
  - For Subtask B: macro-F1 over positive & negative.

We also plot **confusion matrices** to inspect error patterns:

- Rows = **true labels** (ground truth).
- Columns = **predicted labels**.
- Diagonal cells = correct predictions.
- Off-diagonal cells = misclassifications.

For Subtask A, the confusion matrices highlight that:

- Positive and negative classes have relatively high recall.
- The **neutral class is the most challenging**:
  - Many neutral tweets are misclassified as positive or negative,
  - This lowers the macro-average recall even when accuracy looks decent.

---

## 7. Results (Summary)

### 7.1 Subtask A – Tweet-Level Sentiment

Approximate test performance:

| Model          | Accuracy | Macro Recall | F1_PN (pos/neg) |
|----------------|----------|--------------|-----------------|
| BERTweet-base  | ~0.677   | ~0.723       | ~0.699          |
| RoBERTa-base   | slightly lower on all metrics (see report for full table) |

Key observations:

- **BERTweet** (tweet-domain pretraining) outperforms RoBERTa on Subtask A.
- The main bottleneck is **neutral recall**, not positive/negative.

### 7.2 Subtask B – Topic-Based Sentiment (Cross-Encoder)

Approximate test performance:

| Model                   | Accuracy | Macro Recall | Macro-F1 |
|-------------------------|----------|--------------|----------|
| BERTweet-base (CE)      | ~0.928   | ~0.899       | ~0.894   |
| RoBERTa-base (CE)       | ~0.923   | ~0.901       | ~0.888   |

CE = Cross-Encoder (topic + tweet encoded together)

Key observations:

- Both models achieve **very high accuracy and macro recall** despite label imbalance.
- The **cross-encoder architecture** is highly effective for modeling targeted sentiment.

---

## 8. Discussion & Limitations

Some limitations of our current work:

- **Limited hyperparameter search**  
  We use a single, standard set of hyperparameters. More systematic tuning might further improve performance, especially for neutral.

- **No topic-held-out evaluation**  
  For Subtask B, we follow the standard splits. We do not evaluate generalization to completely unseen topics (e.g., leave-one-topic-out experiments).

- **Neutral sentiment remains difficult**  
  Many tweets are ambiguous or weakly emotional. With a simple argmax over softmax, the model tends to choose positive/negative rather than neutral in borderline cases.

- **Limited error analysis**  
  We only inspect a subset of errors, where we see common issues like sarcasm, mixed sentiment, and dependence on external context.

---

## 9. Possible Future Work

Possible extensions and improvements:

- Use **class-balanced loss** or **focal loss** to handle class imbalance (especially neutral/negative).
- Perform **more extensive hyperparameter search** (learning rate, batch size, max length, scheduler).
- Design **topic-held-out** evaluation for Subtask B to test generalization to unseen topics.
- Apply **data augmentation** (e.g. back-translation) for social media text.
- Explore **probability calibration** and **thresholding** for better neutral detection.

---

## 10. Citation

If you use this project or ideas from it, please consider citing the original SemEval shared tasks and the model papers, for example:

- Rosenthal et al., *SemEval-2017 Task 4: Sentiment Analysis in Twitter*.  
- Nguyen et al., *BERTweet: A pre-trained language model for English Tweets*.  
- Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach*.

You can also cite the project report:

> *Tweet-Level and Topic-Based Sentiment Analysis on Twitter with BERTweet and RoBERTa: A SemEval-2017 Task 4 Study*, Deep Learning Final Project, [Your University], [Year].

