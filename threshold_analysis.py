"""
Threshold Analysis Using ROC Curve and Precision-Recall Curve
=============================================================
Add these cells to your notebook AFTER all 5 models are trained
but BEFORE the Ensemble section.

This code:
1. Gets probability predictions from all models
2. Plots ROC curves & finds optimal threshold (Youden's J statistic)
3. Plots PR curves & finds optimal threshold (max F1)
4. Re-evaluates all models using optimal thresholds
5. Compares default (0.5) vs optimal threshold performance
"""

# ============================================================
# CELL 1: Imports & Get Probability Predictions
# ============================================================
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
import matplotlib.pyplot as plt

# Get probability predictions from all models
rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

# SVM: use decision_function (raw scores, not probabilities)
# To get true probabilities, retrain with probability=True:
#   svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
# For now, using decision_function works fine with ROC/PR curves
svm_scores = svm_model.decision_function(X_test_scaled)

lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
nn_prob = nn_model.predict_proba(X_test_scaled)[:, 1]
# cnn_prob already exists from training

print('Probability predictions collected from all models.')

# ============================================================
# CELL 2: ROC Curve Analysis — Find Optimal Threshold
# ============================================================
model_probs = {
    'Random Forest': rf_prob,
    'SVM': svm_scores,
    'Logistic Regression': lr_prob,
    'Neural Network': nn_prob,
    '1D CNN': cnn_prob,
}

roc_results = {}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
colors = ['#2196F3', '#4CAF50', '#9C27B0', '#F44336', '#FF9800']

for i, (name, probs) in enumerate(model_probs.items()):
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    # Youden's J statistic: optimal point = max(TPR - FPR)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    roc_results[name] = {
        'auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'optimal_tpr': tpr[optimal_idx],
        'optimal_fpr': fpr[optimal_idx],
    }

    ax = axes.flatten()[i]
    ax.plot(fpr, tpr, color=colors[i], lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=12,
            label=f'Optimal Threshold = {optimal_threshold:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'{name}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

# Hide the 6th subplot
axes.flatten()[5].set_visible(False)

plt.suptitle("ROC Curves — Optimal Thresholds (Youden's J Statistic)",
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Print ROC results
print('\n' + '='*60)
print('  ROC Curve Analysis — Optimal Thresholds (Youden\'s J)')
print('='*60)
for name, res in roc_results.items():
    print(f'  {name:25s}  AUC={res["auc"]:.4f}  Threshold={res["optimal_threshold"]:.4f}')
print('='*60)


# ============================================================
# CELL 3: Precision-Recall Curve — Find Optimal Threshold
# ============================================================
pr_results = {}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, (name, probs) in enumerate(model_probs.items()):
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)

    # Find threshold that maximizes F1 = 2*P*R / (P+R)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]

    pr_results[name] = {
        'ap': ap,
        'optimal_threshold': optimal_threshold,
        'optimal_f1': optimal_f1,
        'optimal_precision': precision[optimal_idx],
        'optimal_recall': recall[optimal_idx],
    }

    ax = axes.flatten()[i]
    ax.plot(recall, precision, color=colors[i], lw=2, label=f'AP = {ap:.4f}')
    ax.plot(recall[optimal_idx], precision[optimal_idx], 'ro', markersize=12,
            label=f'Threshold={optimal_threshold:.4f} (F1={optimal_f1:.4f})')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(f'{name}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])

axes.flatten()[5].set_visible(False)

plt.suptitle('Precision-Recall Curves — Optimal Thresholds (Max F1)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Print PR results
print('\n' + '='*60)
print('  Precision-Recall Analysis — Optimal Thresholds (Max F1)')
print('='*60)
for name, res in pr_results.items():
    print(f'  {name:25s}  AP={res["ap"]:.4f}  Threshold={res["optimal_threshold"]:.4f}  F1={res["optimal_f1"]:.4f}')
print('='*60)


# ============================================================
# CELL 4: Compare Default (0.5) vs ROC vs PR Thresholds
# ============================================================
print('\n' + '='*70)
print('  THRESHOLD COMPARISON: Default (0.5) vs ROC vs PR')
print('='*70)
print(f'  {"Model":25s} {"Default":>10s} {"ROC (J)":>10s} {"PR (F1)":>10s}')
print('-'*70)
for name in model_probs:
    default = 0.5 if name != 'SVM' else 0.0  # SVM decision_function uses 0 as default
    roc_t = roc_results[name]['optimal_threshold']
    pr_t = pr_results[name]['optimal_threshold']
    print(f'  {name:25s} {default:>10.4f} {roc_t:>10.4f} {pr_t:>10.4f}')
print('='*70)


# ============================================================
# CELL 5: Re-evaluate models with optimal thresholds
# ============================================================
print('\n' + '='*70)
print('  RE-EVALUATION WITH OPTIMAL THRESHOLDS')
print('='*70)

optimal_results = {}

for name, probs in model_probs.items():
    roc_thresh = roc_results[name]['optimal_threshold']
    pr_thresh = pr_results[name]['optimal_threshold']

    # Predictions with ROC optimal threshold
    roc_pred = (probs >= roc_thresh).astype(int)
    roc_f1 = f1_score(y_test, roc_pred)
    roc_acc = accuracy_score(y_test, roc_pred)

    # Predictions with PR optimal threshold
    pr_pred = (probs >= pr_thresh).astype(int)
    pr_f1 = f1_score(y_test, pr_pred)
    pr_acc = accuracy_score(y_test, pr_pred)

    # Default threshold predictions (already computed)
    if name == 'SVM':
        default_pred = svm_pred
    elif name == 'Random Forest':
        default_pred = rf_pred
    elif name == 'Logistic Regression':
        default_pred = lr_pred
    elif name == 'Neural Network':
        default_pred = nn_pred
    elif name == '1D CNN':
        default_pred = cnn_pred

    default_f1 = f1_score(y_test, default_pred)
    default_acc = accuracy_score(y_test, default_pred)

    optimal_results[name] = {
        'default_f1': default_f1, 'default_acc': default_acc,
        'roc_f1': roc_f1, 'roc_acc': roc_acc, 'roc_thresh': roc_thresh,
        'pr_f1': pr_f1, 'pr_acc': pr_acc, 'pr_thresh': pr_thresh,
    }

# Print comparison table
print(f'\n  {"Model":25s} | {"Default F1":>12s} | {"ROC F1":>12s} | {"PR F1":>12s} | {"Best Method":>12s}')
print('-'*90)
for name, res in optimal_results.items():
    best = 'Default'
    best_f1 = res['default_f1']
    if res['roc_f1'] > best_f1:
        best = 'ROC'
        best_f1 = res['roc_f1']
    if res['pr_f1'] > best_f1:
        best = 'PR'

    print(f'  {name:25s} | {res["default_f1"]:>12.4f} | {res["roc_f1"]:>12.4f} | {res["pr_f1"]:>12.4f} | {best:>12s}')
print('='*90)

print(f'\n  {"Model":25s} | {"Default Acc":>12s} | {"ROC Acc":>12s} | {"PR Acc":>12s}')
print('-'*75)
for name, res in optimal_results.items():
    print(f'  {name:25s} | {res["default_acc"]:>12.4f} | {res["roc_acc"]:>12.4f} | {res["pr_acc"]:>12.4f}')
print('='*75)


# ============================================================
# CELL 6: Final Summary — Recommended Thresholds
# ============================================================
print('\n' + '='*70)
print('  RECOMMENDED THRESHOLDS')
print('='*70)
for name, res in optimal_results.items():
    scores = {
        'Default (0.5)': res['default_f1'],
        f'ROC ({res["roc_thresh"]:.4f})': res['roc_f1'],
        f'PR ({res["pr_thresh"]:.4f})': res['pr_f1'],
    }
    best_method = max(scores, key=scores.get)
    best_f1 = scores[best_method]
    print(f'  {name:25s} → {best_method} → F1={best_f1:.4f}')
print('='*70)
