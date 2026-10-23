# DeepShield: CNN, RNN, LSTM & GRU for Network Traffic Anomaly Detection

A complete student CA project that trains four deep-learning architectures to classify network-flow rows as **Benign** or **Attack**, evaluates them on a held-out test split, and provides a Streamlit demo for CSV predictions.

## 1. Tech stack
- Python, TensorFlow/Keras
- NumPy, pandas, scikit-learn
- Streamlit dashboard
- Dataset: CIC-IDS2017 from the Canadian Institute for Cybersecurity

Official dataset: https://www.unb.ca/cic/datasets/ids-2017.html

## 2. Folder structure
```text
deep_learning_ids_ca/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── README.md
└── artifacts/        # generated after training; not committed
```

## 3. Setup
Python 3.10 or 3.11 is recommended. Create a virtual environment:

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows
```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Dataset
1. Open the official CIC-IDS2017 page: https://www.unb.ca/cic/datasets/ids-2017.html
2. Download the machine-learning CSV archive (often named `MachineLearningCSV.zip`).
3. Extract the CSV files into the project's `data/` folder.
4. Check that the CSVs contain a `Label` column with `BENIGN` and attack labels.

The original dataset is large. This starter project caps the default training sample at 60,000 rows so it is more practical for a CA. Increase `--max-rows` if your machine can handle it.

## 5. Train all four models
```bash
python train.py --epochs 8 --max-rows 60000
```
This trains CNN, SimpleRNN, LSTM and GRU models and writes:
- `artifacts/cnn_model.keras`
- `artifacts/rnn_model.keras`
- `artifacts/lstm_model.keras`
- `artifacts/gru_model.keras`
- `artifacts/scaler.joblib`
- `artifacts/features.json`
- `artifacts/metrics.json`

For a quick smoke test, use `--max-rows 5000 --epochs 2`. Do not use smoke-test metrics as final results.

## 6. Launch dashboard
After training:
```bash
streamlit run app.py
```
Upload a CSV containing the same feature columns as the training dataset. The dashboard displays model choice, attack probability, predicted label, and a downloadable predictions CSV.

## 7. Methodology
1. Load and combine labeled network-flow CSVs.
2. Clean column names, coerce numeric features, replace infinity and missing values.
3. Map `BENIGN` to class 0 and all other labels to class 1.
4. Use stratified train/validation/test split (70% / 15% / 15%).
5. Fit StandardScaler on the training split only, then transform validation and test splits.
6. Train each architecture with binary cross-entropy and Adam; early stopping uses validation loss.
7. Evaluate on the held-out test set with accuracy, precision, recall, F1 and confusion matrix.
8. Save models and launch a dashboard for inference.

### Important limitation to mention in viva
This is a **binary classification baseline**, not a production-grade intrusion prevention system. Each CSV row is treated as an ordered feature vector for the neural network. Although RNN/LSTM/GRU layers are sequence-oriented, the features are not a true time-window sequence of network events in this baseline. For a research extension, group flows by time and host and build fixed-length temporal windows; use time/day-based splitting to reduce leakage. Dataset label imbalance and duplicate/invalid rows can also affect reported metrics.

## 8. Model intuition
- **CNN:** Conv1D learns local patterns across adjacent numeric features.
- **RNN:** Simple recurrent layer processes the feature vector step by step; compact baseline, but may struggle with long dependencies.
- **LSTM:** Uses input/forget/output gates and a cell state to preserve longer dependencies.
- **GRU:** Uses update/reset gates; often fewer parameters than LSTM.

## 9. CA report outline
1. Title page
2. Abstract
3. Problem statement and objectives
4. Literature review / background on CNN, RNN, LSTM, GRU
5. Dataset description and citation
6. System architecture and methodology
7. Implementation and screenshots
8. Results table (fill with the generated `metrics.json` values)
9. Discussion, limitations and future scope
10. Conclusion and references

## 10. Suggested viva questions
1. What is deep learning?
2. How is CNN different from RNN?
3. Why do LSTM and GRU use gates?
4. What is binary cross-entropy?
5. Why do we scale numeric features?
6. Why is the dataset split into train, validation and test sets?
7. What is precision vs recall?
8. Why can accuracy be misleading for imbalanced datasets?
9. What does a confusion matrix show?
10. Why should test data not be used during training?

## 11. References
- Canadian Institute for Cybersecurity, *Intrusion Detection Evaluation Dataset (CIC-IDS2017)*: https://www.unb.ca/cic/datasets/ids-2017.html
- TensorFlow, *Working with RNNs*: https://www.tensorflow.org/guide/keras/working_with_rnns
