"""
Score Threshold Analysis — All Models (Including 1D CNN & Ensemble)
===================================================================
Tests multiple Score thresholds with ALL 9 models using 5-Fold CV
to find the threshold that gives the best overall performance.
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.base import clone
from xgboost import XGBClassifier
from scipy.stats import mode
from collections import Counter

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# CELL 1: Load & Preprocess Data
# ============================================================
df_raw = pd.read_csv('Malware_Analysis.csv')

columns_to_drop = [
    'pid', 'info_id',
    'API_GetAdaptersInfo', 'API_InternetSetStatusCallback',
    'API_SetFileInformationByHandle', 'API_getaddrinfo',
    'API_RtlDecompressBuffer', 'API_DnsQuery_A',
    'API_Module32NextW', 'API_ControlService',
    'API_JsGlobalObjectDefaultEvalHelper',
    'API_Thread32Next', 'API_Thread32First',
    'ttp_T1045_short', 'ttp_T1045_long',
    'ttp_T1082_short', 'ttp_T1082_long',
    'ttp_T1158_short', 'ttp_T1158_long',
    'description_y', 'description_x', 'references'
]

df_temp = df_raw.drop(columns=columns_to_drop, errors='ignore')

cat_cols = df_temp.select_dtypes(include=['object']).columns.tolist()
for c in cat_cols:
    le = LabelEncoder()
    df_temp[c] = df_temp[c].fillna('missing')
    df_temp[c] = le.fit_transform(df_temp[c])
df_temp = df_temp.fillna(0)

print(f'Dataset: {df_temp.shape[0]} samples, {df_temp.shape[1]} features')
print(f'Score range: {df_temp["Score"].min()} to {df_temp["Score"].max()}')


# ============================================================
# CELL 2: Define sklearn Models + CNN builder
# ============================================================
sklearn_models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20,
        min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5,
        learning_rate=0.1, subsample=0.8, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=5,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, use_label_encoder=False),
}

def build_cnn(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

all_model_names = list(sklearn_models.keys()) + ['1D CNN', 'Ensemble']
print(f'Models to test: {all_model_names}')


# ============================================================
# CELL 3: Test All Thresholds with All Models (5-Fold CV)
# ============================================================
thresholds = [3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_results = []

total = len(thresholds) * (len(sklearn_models) + 2)
print(f'\nTesting {len(thresholds)} thresholds x {len(sklearn_models) + 2} models = {total} experiments')
print('This will take ~20-30 minutes (CNN + Ensemble are slow)...\n')

for thresh in thresholds:
    y = (df_temp['Score'] >= thresh).astype(int)
    X = df_temp.drop(columns=['Score'])
    malware_pct = y.mean() * 100

    if malware_pct < 10 or malware_pct > 90:
        print(f'  Threshold {thresh:.1f}: SKIPPED (imbalanced: {malware_pct:.1f}%)')
        continue

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f'  Threshold {thresh:.1f} (Malware: {malware_pct:.1f}%)')

    # --- sklearn models ---
    for model_name, model in sklearn_models.items():
        start = time.time()
        f1_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
        acc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        elapsed = time.time() - start

        all_results.append({
            'threshold': thresh,
            'model': model_name,
            'f1_mean': f1_scores.mean(),
            'f1_std': f1_scores.std(),
            'acc_mean': acc_scores.mean(),
            'acc_std': acc_scores.std(),
            'malware_pct': malware_pct,
            'time': elapsed,
        })
        print(f'    {model_name:25s} F1={f1_scores.mean():.4f}  Acc={acc_scores.mean():.4f}  ({elapsed:.1f}s)')

    # --- 1D CNN (manual 5-Fold CV) ---
    cnn_f1_list = []
    cnn_acc_list = []
    start = time.time()

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y)):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y.values[train_idx], y.values[test_idx]

        X_tr_cnn = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
        X_te_cnn = X_te.reshape(X_te.shape[0], X_te.shape[1], 1)

        cnn = build_cnn((X_tr.shape[1], 1))
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
        cnn.fit(X_tr_cnn, y_tr, epochs=50, batch_size=32, validation_split=0.2,
                callbacks=[early_stop], verbose=0)

        cnn_prob = cnn.predict(X_te_cnn, verbose=0).flatten()
        cnn_pred = (cnn_prob >= 0.5).astype(int)
        cnn_f1_list.append(f1_score(y_te, cnn_pred))
        cnn_acc_list.append(accuracy_score(y_te, cnn_pred))

    cnn_elapsed = time.time() - start
    all_results.append({
        'threshold': thresh,
        'model': '1D CNN',
        'f1_mean': np.mean(cnn_f1_list),
        'f1_std': np.std(cnn_f1_list),
        'acc_mean': np.mean(cnn_acc_list),
        'acc_std': np.std(cnn_acc_list),
        'malware_pct': malware_pct,
        'time': cnn_elapsed,
    })
    print(f'    {"1D CNN":25s} F1={np.mean(cnn_f1_list):.4f}  Acc={np.mean(cnn_acc_list):.4f}  ({cnn_elapsed:.1f}s)')

    # --- Ensemble (manual 5-Fold CV, majority vote of all 8 models) ---
    ens_f1_list = []
    ens_acc_list = []
    start = time.time()

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y)):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y.values[train_idx], y.values[test_idx]

        fold_preds = []

        # Train all 7 sklearn models
        for m in sklearn_models.values():
            m_clone = clone(m)
            m_clone.fit(X_tr, y_tr)
            fold_preds.append(m_clone.predict(X_te))

        # Train 1D CNN
        X_tr_cnn = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
        X_te_cnn = X_te.reshape(X_te.shape[0], X_te.shape[1], 1)
        cnn = build_cnn((X_tr.shape[1], 1))
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
        cnn.fit(X_tr_cnn, y_tr, epochs=50, batch_size=32, validation_split=0.2,
                callbacks=[early_stop], verbose=0)
        cnn_prob = cnn.predict(X_te_cnn, verbose=0).flatten()
        fold_preds.append((cnn_prob >= 0.5).astype(int))

        # Majority vote across all 8 models
        all_preds_arr = np.array(fold_preds)
        ens_pred, _ = mode(all_preds_arr, axis=0, keepdims=False)
        ens_pred = ens_pred.flatten()

        ens_f1_list.append(f1_score(y_te, ens_pred))
        ens_acc_list.append(accuracy_score(y_te, ens_pred))

    ens_elapsed = time.time() - start
    all_results.append({
        'threshold': thresh,
        'model': 'Ensemble',
        'f1_mean': np.mean(ens_f1_list),
        'f1_std': np.std(ens_f1_list),
        'acc_mean': np.mean(ens_acc_list),
        'acc_std': np.std(ens_acc_list),
        'malware_pct': malware_pct,
        'time': ens_elapsed,
    })
    print(f'    {"Ensemble":25s} F1={np.mean(ens_f1_list):.4f}  Acc={np.mean(ens_acc_list):.4f}  ({ens_elapsed:.1f}s)')

results_df = pd.DataFrame(all_results)


# ============================================================
# CELL 4: Results Table
# ============================================================
pivot_f1 = results_df.pivot(index='threshold', columns='model', values='f1_mean')
pivot_f1 = pivot_f1[all_model_names]

print('\n' + '='*120)
print('  F1-SCORE: Each Model x Each Threshold (5-Fold CV)')
print('='*120)
print(pivot_f1.round(4).to_string())

pivot_acc = results_df.pivot(index='threshold', columns='model', values='acc_mean')
pivot_acc = pivot_acc[all_model_names]

print('\n' + '='*120)
print('  ACCURACY: Each Model x Each Threshold (5-Fold CV)')
print('='*120)
print(pivot_acc.round(4).to_string())


# ============================================================
# CELL 5: Best Threshold Per Model + Majority Voting
# ============================================================
print('\n' + '='*70)
print('  BEST THRESHOLD PER MODEL (by F1-Score)')
print('='*70)

best_per_model = {}
for model_name in all_model_names:
    model_data = results_df[results_df['model'] == model_name]
    best_idx = model_data['f1_mean'].idxmax()
    best_row = model_data.loc[best_idx]
    best_per_model[model_name] = best_row['threshold']
    print(f'  {model_name:25s} -> Threshold = {best_row["threshold"]:.1f}  (F1={best_row["f1_mean"]:.4f})')

votes = Counter(best_per_model.values())
print(f'\n  MAJORITY VOTING:')
for thresh, count in votes.most_common():
    model_list = [m for m, t in best_per_model.items() if t == thresh]
    print(f'    Score >= {thresh:.1f} : {count} vote(s) from {model_list}')

winner = votes.most_common(1)[0][0]
print(f'\n  >> WINNER (majority vote): Score >= {winner:.1f}')
print('='*70)


# ============================================================
# CELL 6: Average F1 Across All 9 Models Per Threshold
# ============================================================
avg_per_threshold = results_df.groupby('threshold').agg(
    avg_f1=('f1_mean', 'mean'),
    avg_acc=('acc_mean', 'mean'),
    min_f1=('f1_mean', 'min'),
    max_f1=('f1_mean', 'max'),
).reset_index()

print('\n' + '='*70)
print('  AVERAGE PERFORMANCE ACROSS ALL 9 MODELS')
print('='*70)
print(f'  {"Threshold":>10s} | {"Avg F1":>10s} | {"Avg Acc":>10s} | {"Min F1":>10s} | {"Max F1":>10s}')
print(f'  {"-"*55}')
for _, row in avg_per_threshold.iterrows():
    print(f'  {row["threshold"]:>10.1f} | {row["avg_f1"]:>10.4f} | {row["avg_acc"]:>10.4f} | {row["min_f1"]:>10.4f} | {row["max_f1"]:>10.4f}')

best_avg = avg_per_threshold.loc[avg_per_threshold['avg_f1'].idxmax()]
print(f'\n  >> BEST THRESHOLD (highest avg F1 across all 9 models): {best_avg["threshold"]:.1f}')
print(f'     Avg F1={best_avg["avg_f1"]:.4f}, Avg Acc={best_avg["avg_acc"]:.4f}')
print('='*70)


# ============================================================
# CELL 7: Plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
colors = ['#2196F3', '#4CAF50', '#9C27B0', '#F44336', '#00BCD4',
          '#795548', '#E91E63', '#FF9800', '#607D8B']

for i, model_name in enumerate(all_model_names):
    model_data = results_df[results_df['model'] == model_name]
    axes[0].plot(model_data['threshold'], model_data['f1_mean'], 'o-',
                 color=colors[i], linewidth=2, markersize=6, label=model_name)

axes[0].axvline(x=best_avg['threshold'], color='black', linestyle='--', linewidth=2,
                label=f'Best = {best_avg["threshold"]:.1f}')
axes[0].set_xlabel('Score Threshold', fontsize=12)
axes[0].set_ylabel('F1-Score (5-Fold CV)', fontsize=12)
axes[0].set_title('F1-Score vs Score Threshold (All 9 Models)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=8, loc='best')
axes[0].grid(alpha=0.3)

# Average F1 bar chart
bar_colors = ['#FF9800' if t == best_avg['threshold'] else '#2196F3'
              for t in avg_per_threshold['threshold']]
axes[1].bar(avg_per_threshold['threshold'].astype(str), avg_per_threshold['avg_f1'],
            color=bar_colors, edgecolor='white', alpha=0.9)
for i, row in avg_per_threshold.iterrows():
    axes[1].text(i, row['avg_f1'] + 0.002, f'{row["avg_f1"]:.4f}',
                 ha='center', fontsize=9, fontweight='bold')
axes[1].set_xlabel('Score Threshold', fontsize=12)
axes[1].set_ylabel('Average F1-Score', fontsize=12)
axes[1].set_title('Average F1 Across All 9 Models (Orange = Best)', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('threshold_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# CELL 8: Final Recommendation
# ============================================================
print('\n' + '='*70)
print('  FINAL RECOMMENDATION')
print('='*70)
print(f'\n  Method 1 (Best Avg F1 across all 9 models):  Score >= {best_avg["threshold"]:.1f}')
print(f'  Method 2 (Majority vote by best per model):   Score >= {winner:.1f}')
print(f'\n  Use in your notebook:')
print(f'     df["label"] = (df["Score"] >= {best_avg["threshold"]:.1f}).astype(int)')
print('='*70)
