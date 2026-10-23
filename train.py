import os, glob, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras import layers, models, callbacks

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def build_model(kind, input_shape):
    model = models.Sequential(name=kind.lower() + "_ids")
    model.add(layers.Input(shape=input_shape))
    if kind == "CNN":
        model.add(layers.Conv1D(32, 3, padding="same", activation="relu"))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Conv1D(64, 3, padding="same", activation="relu"))
        model.add(layers.GlobalMaxPooling1D())
    elif kind == "RNN":
        model.add(layers.SimpleRNN(48, dropout=0.15))
    elif kind == "LSTM":
        model.add(layers.LSTM(48, dropout=0.15))
    elif kind == "GRU":
        model.add(layers.GRU(48, dropout=0.15))
    else:
        raise ValueError("Unknown architecture")
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def load_data(data_dir, max_rows, sample_seed):
    paths = glob.glob(os.path.join(data_dir, "*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. See data/README.md.")
    frames = []
    for path in paths:
        print(f"Reading {os.path.basename(path)}")
        try:
            df = pd.read_csv(path, low_memory=False, encoding_errors="replace")
        except TypeError:
            df = pd.read_csv(path, low_memory=False)
        df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
        label_candidates = [c for c in df.columns if c.lower() in ("label", "class", "target", "attack")]
        if not label_candidates:
            raise ValueError(f"Could not find a Label/Class/Target column in {path}")
        label_col = label_candidates[0]
        df = df.rename(columns={label_col: "Label"})
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data.columns = [str(c).strip() for c in data.columns]
    data = data.loc[:, ~data.columns.duplicated()]
    data["Label"] = data["Label"].astype(str).str.strip()
    data = data[data["Label"].str.lower().ne("nan")]
    data["target"] = (data["Label"].str.lower() != "benign").astype("int8")
    # Remove non-feature identifiers and original text label.
    drop_cols = [c for c in data.columns if c.lower() in
                 ("label", "target", "flow id", "source ip", "destination ip", "timestamp")]
    X = data.drop(columns=drop_cols, errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    # Remove constant columns.
    X = X.loc[:, X.nunique(dropna=False) > 1]
    y = data["target"].astype("int8")
    # Balanced cap to keep CA training manageable while preserving both classes.
    if max_rows and len(X) > max_rows:
        per_class = max(1, max_rows // 2)
        idx = []
        for cls in (0, 1):
            candidates = y[y == cls].index
            take = min(len(candidates), per_class)
            idx.extend(pd.Series(candidates).sample(n=take, random_state=sample_seed).tolist())
        # Fill remaining slots from any unused rows.
        if len(idx) < max_rows:
            remaining = np.setdiff1d(np.arange(len(X)), np.asarray(idx, dtype=int))
            extra = min(max_rows - len(idx), len(remaining))
            idx.extend(np.random.default_rng(sample_seed).choice(remaining, size=extra, replace=False).tolist())
        idx = sorted(idx)
        X, y = X.iloc[idx], y.iloc[idx]
    print(f"Rows used: {len(X):,}; numeric features: {X.shape[1]}; attack rate: {y.mean():.3f}")
    if y.nunique() < 2:
        raise ValueError("Dataset must contain both Benign and attack rows.")
    return X, y

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--max-rows", type=int, default=60000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--output-dir", default="artifacts")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    X, y = load_data(args.data_dir, args.max_rows, SEED)
    feature_names = list(X.columns)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    # Interpret each flow's ordered numeric feature vector as a 1D sequence.
    X_train = X_train.astype("float32")[..., np.newaxis]
    X_val = X_val.astype("float32")[..., np.newaxis]
    X_test = X_test.astype("float32")[..., np.newaxis]
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))
    with open(os.path.join(args.output_dir, "features.json"), "w") as f:
        json.dump(feature_names, f, indent=2)
    results = []
    for kind in ["CNN", "RNN", "LSTM", "GRU"]:
        print(f"\n{'='*20} Training {kind} {'='*20}")
        tf.keras.backend.clear_session()
        model = build_model(kind, (X_train.shape[1], 1))
        early = callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=args.epochs, batch_size=args.batch_size,
                            callbacks=[early], verbose=1)
        probs = model.predict(X_test, batch_size=args.batch_size, verbose=0).ravel()
        pred = (probs >= 0.5).astype(int)
        metrics = {
            "model": kind,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "epochs_trained": len(history.history["loss"])
        }
        print(json.dumps(metrics, indent=2))
        model.save(os.path.join(args.output_dir, f"{kind.lower()}_model.keras"))
        results.append(metrics)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nTraining complete. Models and metrics saved in:", args.output_dir)
    print("\nReport these measured values; do not invent accuracy numbers.")

if __name__ == "__main__":
    main()
