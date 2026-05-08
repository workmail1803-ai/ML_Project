import json

# Load notebook
with open('main_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find the insertion point: after Cell 21 (1D CNN code) and before Cell 22 (Ensemble markdown)
# We need to find the "## 11. Ensemble" markdown cell
insert_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'markdown' and c['source'] and '11. Ensemble' in ''.join(c['source']):
        insert_idx = i
        break

if insert_idx is None:
    print("Could not find Ensemble markdown cell!")
    exit(1)

print(f"Found Ensemble markdown at cell index {insert_idx}")
print(f"Will insert 6 new cells (3 markdown + 3 code) before it")

# Create new cells
new_cells = [
    # AdaBoost markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 11. AdaBoost"]
    },
    # AdaBoost code
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier\n",
            "from xgboost import XGBClassifier\n",
            "\n",
            "start = time.time()\n",
            "ada_model = AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=42)\n",
            "ada_model.fit(X_train_scaled, y_train)\n",
            "ada_time = time.time() - start\n",
            "ada_pred = ada_model.predict(X_test_scaled)\n",
            "evaluate_model('AdaBoost', y_test, ada_pred, ada_time)"
        ]
    },
    # Gradient Boosting markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 12. Gradient Boosting"]
    },
    # Gradient Boosting code
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "start = time.time()\n",
            "gb_model = GradientBoostingClassifier(n_estimators=200, max_depth=5,\n",
            "    learning_rate=0.1, subsample=0.8, random_state=42)\n",
            "gb_model.fit(X_train_scaled, y_train)\n",
            "gb_time = time.time() - start\n",
            "gb_pred = gb_model.predict(X_test_scaled)\n",
            "evaluate_model('Gradient Boosting', y_test, gb_pred, gb_time)"
        ]
    },
    # XGBoost markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 13. XGBoost"]
    },
    # XGBoost code
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "start = time.time()\n",
            "xgb_model = XGBClassifier(n_estimators=200, max_depth=5,\n",
            "    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,\n",
            "    eval_metric='logloss', random_state=42, use_label_encoder=False)\n",
            "xgb_model.fit(X_train_scaled, y_train)\n",
            "xgb_time = time.time() - start\n",
            "xgb_pred = xgb_model.predict(X_test_scaled)\n",
            "evaluate_model('XGBoost', y_test, xgb_pred, xgb_time)"
        ]
    },
]

# Insert new cells
for j, cell in enumerate(new_cells):
    cells.insert(insert_idx + j, cell)

# Now update the section numbers for cells after the insertion
# The old "11. Ensemble" becomes "14. Ensemble"
# Update "12. Model Comparison" -> "15. Model Comparison"
# etc.
renumber_map = {
    '## 11. Ensemble': '## 14. Ensemble (Majority Voting)',
    '## 12. Model Comparison': '## 15. Model Comparison Table',
    '## 13. Confusion Matrices': '## 16. Confusion Matrices',
    '## 14. Bar Chart': '## 17. Bar Chart',
    '## 14. Final': '## 18. Final',
}

for c in cells:
    if c['cell_type'] == 'markdown' and c['source']:
        src = ''.join(c['source']).strip()
        for old, new in renumber_map.items():
            if old in src:
                c['source'] = [new]
                break

# Update the Ensemble code cell to include all 8 models
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and c['source']:
        src = ''.join(c['source'])
        if 'from scipy.stats import mode' in src and 'ensemble_pred' in src:
            c['source'] = [
                "from scipy.stats import mode\n",
                "import time\n",
                "import numpy as np\n",
                "\n",
                "start = time.time()\n",
                "# Combine predictions from all 8 models\n",
                "all_preds = np.array([rf_pred, svm_pred, lr_pred, nn_pred, cnn_pred, ada_pred, gb_pred, xgb_pred])\n",
                "# Perform hard voting (majority vote)\n",
                "ensemble_pred, _ = mode(all_preds, axis=0, keepdims=False)\n",
                "ensemble_pred = ensemble_pred.flatten()\n",
                "ensemble_time = time.time() - start\n",
                "\n",
                "evaluate_model('Ensemble', y_test, ensemble_pred, ensemble_time)"
            ]
            print(f"  Updated Ensemble cell at index {i}")
            break

# Update model_names in comparison cell
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and c['source']:
        src = ''.join(c['source'])
        if 'model_names' in src and 'comparison_data' in src:
            c['source'] = [
                "model_names = ['Random Forest', 'SVM', 'Logistic Regression', 'Neural Network', '1D CNN',\n",
                "               'AdaBoost', 'Gradient Boosting', 'XGBoost', 'Ensemble']\n",
                "comparison_data = []\n",
                "for name in model_names:\n",
                "    r = results[name]\n",
                "    comparison_data.append({\n",
                "        'Model': name,\n",
                "        'Accuracy': f\"{r['Accuracy']:.4f}\",\n",
                "        'Precision': f\"{r['Precision']:.4f}\",\n",
                "        'Recall': f\"{r['Recall']:.4f}\",\n",
                "        'F1-Score': f\"{r['F1-Score']:.4f}\",\n",
                "        'Time (s)': f\"{r['Training Time (s)']:.4f}\"\n",
                "    })\n",
                "comparison_df = pd.DataFrame(comparison_data)\n",
                "print('MODEL COMPARISON SUMMARY')\n",
                "print(comparison_df.to_string(index=False))"
            ]
            print(f"  Updated comparison cell at index {i}")
            break

# Update confusion matrix cell
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and c['source']:
        src = ''.join(c['source'])
        if 'plt.subplots(2, 3' in src and 'heatmap' in src:
            c['source'] = [
                "fig, axes = plt.subplots(3, 3, figsize=(18, 15))\n",
                "colors = ['Blues', 'Greens', 'Purples', 'Reds', 'Oranges', 'YlGn', 'BuPu', 'RdYlBu', 'Greys']\n",
                "for i, (name, cmap) in enumerate(zip(model_names, colors)):\n",
                "    cm = confusion_matrix(y_test, results[name]['y_pred'])\n",
                "    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes.flatten()[i],\n",
                "        xticklabels=['Benign', 'Malware'], yticklabels=['Benign', 'Malware'],\n",
                "        annot_kws={'size': 14})\n",
                "    axes.flatten()[i].set_title(name, fontsize=14, fontweight='bold')\n",
                "    axes.flatten()[i].set_ylabel('Actual')\n",
                "    axes.flatten()[i].set_xlabel('Predicted')\n",
                "plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
            print(f"  Updated confusion matrix cell at index {i}")
            break

# Update bar chart cell
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and c['source']:
        src = ''.join(c['source'])
        if "metrics = ['Accuracy'" in src and 'bar_colors' in src:
            c['source'] = [
                "metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']\n",
                "x = np.arange(len(metrics))\n",
                "width = 0.09\n",
                "bar_colors = ['#2196F3', '#4CAF50', '#9C27B0', '#F44336', '#FF9800',\n",
                "              '#00BCD4', '#795548', '#E91E63', '#607D8B']\n",
                "fig, ax = plt.subplots(figsize=(16, 7))\n",
                "for i, j in enumerate(model_names):\n",
                "    values = [results[j][m] for m in metrics]\n",
                "    bars = ax.bar(x + i * width, values, width, label=j,\n",
                "        color=bar_colors[i], edgecolor='white', linewidth=0.5)\n",
                "    for bar, val in zip(bars, values):\n",
                "        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,\n",
                "            f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')\n",
                "ax.set_xlabel('Metrics', fontsize=12)\n",
                "ax.set_ylabel('Score', fontsize=12)\n",
                "ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')\n",
                "ax.set_xticks(x + 4 * width)\n",
                "ax.set_xticklabels(metrics, fontsize=11)\n",
                "ax.legend(fontsize=9, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.08))\n",
                "ax.set_ylim(0, 1.12)\n",
                "ax.grid(axis='y', alpha=0.3)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
            print(f"  Updated bar chart cell at index {i}")
            break

# Save
with open('main_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nDone! Notebook now has {len(cells)} cells (was 34)")
print("New cells added: AdaBoost, Gradient Boosting, XGBoost")
print("Updated cells: Ensemble, Comparison, Confusion Matrix, Bar Chart")
