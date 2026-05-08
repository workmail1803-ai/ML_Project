import json

with open('main_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find the "## 3. Create Target Label" markdown cell (currently at index 5)
insert_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'markdown' and c['source']:
        src = ''.join(c['source'])
        if '3. Create Target Label' in src:
            insert_idx = i
            break

if insert_idx is None:
    print("ERROR: Could not find '3. Create Target Label' cell")
    exit(1)

print(f"Inserting threshold analysis before cell {insert_idx}")

# Create the threshold analysis cells
threshold_cells = [
    # --- Markdown: Section header ---
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Score Distribution Analysis & Threshold Justification\n",
            "\n",
            "Before creating binary labels, we analyze the Score distribution to determine\n",
            "the optimal threshold. We test multiple candidate thresholds using 5-Fold\n",
            "Cross-Validation with all models and use majority voting to select the best one."
        ]
    },
    # --- Code: Score Distribution Plot ---
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Score Distribution Analysis ---\n",
            "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n",
            "\n",
            "# Histogram with candidate thresholds\n",
            "axes[0].hist(df['Score'], bins=50, color='#2196F3', edgecolor='white', alpha=0.8)\n",
            "axes[0].set_xlabel('Score', fontsize=12)\n",
            "axes[0].set_ylabel('Count', fontsize=12)\n",
            "axes[0].set_title('Distribution of Malware Scores', fontsize=14, fontweight='bold')\n",
            "for t, color, ls in [(5.0, 'red', '--'), (6.0, 'orange', '--'), (6.5, 'green', '--'), (7.0, 'purple', '--')]:\n",
            "    axes[0].axvline(x=t, color=color, linestyle=ls, linewidth=2, label=f'Threshold = {t}')\n",
            "axes[0].legend(fontsize=10)\n",
            "axes[0].grid(alpha=0.3)\n",
            "\n",
            "# Class balance at each threshold\n",
            "candidate_thresholds = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0]\n",
            "malware_counts = [(df['Score'] >= t).sum() for t in candidate_thresholds]\n",
            "benign_counts  = [(df['Score'] <  t).sum() for t in candidate_thresholds]\n",
            "x_pos = np.arange(len(candidate_thresholds))\n",
            "axes[1].bar(x_pos - 0.2, benign_counts, 0.4, label='Benign', color='#4CAF50')\n",
            "axes[1].bar(x_pos + 0.2, malware_counts, 0.4, label='Malware', color='#F44336')\n",
            "axes[1].set_xticks(x_pos)\n",
            "axes[1].set_xticklabels([str(t) for t in candidate_thresholds])\n",
            "axes[1].set_xlabel('Score Threshold', fontsize=12)\n",
            "axes[1].set_ylabel('Sample Count', fontsize=12)\n",
            "axes[1].set_title('Class Balance at Each Threshold', fontsize=14, fontweight='bold')\n",
            "axes[1].legend(fontsize=11)\n",
            "axes[1].grid(axis='y', alpha=0.3)\n",
            "for i, (b, m) in enumerate(zip(benign_counts, malware_counts)):\n",
            "    axes[1].text(i - 0.2, b + 10, str(b), ha='center', fontsize=9, fontweight='bold')\n",
            "    axes[1].text(i + 0.2, m + 10, str(m), ha='center', fontsize=9, fontweight='bold')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print('Score Statistics:')\n",
            "print(df['Score'].describe())\n",
            "print(f'\\nGap Analysis: Only {((df[\"Score\"] > 5.0) & (df[\"Score\"] < 7.0)).sum()} samples have Score between 5.0 and 7.0')\n",
            "print('This natural gap means thresholds between 5-7 will produce nearly identical results.')"
        ]
    },
    # --- Markdown: Threshold Testing ---
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Testing Each Threshold with All Models (5-Fold Cross-Validation)\n",
            "\n",
            "We test thresholds 4.0, 5.0, 5.5, 6.0, 6.5, and 7.0 with 7 classifiers.\n",
            "For each threshold, we create binary labels and evaluate using Stratified 5-Fold CV."
        ]
    },
    # --- Code: Threshold Testing with All Models ---
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.model_selection import cross_val_score, StratifiedKFold\n",
            "from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.neural_network import MLPClassifier\n",
            "from xgboost import XGBClassifier\n",
            "\n",
            "# Prepare features for threshold testing\n",
            "df_thresh_test = df.copy()\n",
            "cat_cols_test = df_thresh_test.select_dtypes(include=['object']).columns.tolist()\n",
            "for c in cat_cols_test:\n",
            "    le = LabelEncoder()\n",
            "    df_thresh_test[c] = df_thresh_test[c].fillna('missing')\n",
            "    df_thresh_test[c] = le.fit_transform(df_thresh_test[c])\n",
            "df_thresh_test = df_thresh_test.fillna(0)\n",
            "\n",
            "# Define all models to test\n",
            "test_models = {\n",
            "    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20,\n",
            "        min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1),\n",
            "    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),\n",
            "    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),\n",
            "    'Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),\n",
            "    'AdaBoost': AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=42),\n",
            "    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5,\n",
            "        learning_rate=0.1, subsample=0.8, random_state=42),\n",
            "    'XGBoost': XGBClassifier(n_estimators=200, max_depth=5,\n",
            "        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,\n",
            "        eval_metric='logloss', random_state=42, use_label_encoder=False),\n",
            "}\n",
            "\n",
            "candidate_thresholds = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0]\n",
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
            "threshold_results = []\n",
            "\n",
            "print(f'Testing {len(candidate_thresholds)} thresholds x {len(test_models)} models...')\n",
            "print(f'{\"Threshold\":>10s} | {\"Model\":>25s} | {\"F1-Score\":>10s} | {\"Accuracy\":>10s}')\n",
            "print('-' * 65)\n",
            "\n",
            "for thresh in candidate_thresholds:\n",
            "    y_temp = (df_thresh_test['Score'] >= thresh).astype(int)\n",
            "    X_temp = df_thresh_test.drop(columns=['Score'])\n",
            "    scaler_temp = StandardScaler()\n",
            "    X_temp_scaled = scaler_temp.fit_transform(X_temp)\n",
            "\n",
            "    for model_name, model in test_models.items():\n",
            "        f1_scores = cross_val_score(model, X_temp_scaled, y_temp, cv=cv, scoring='f1')\n",
            "        acc_scores = cross_val_score(model, X_temp_scaled, y_temp, cv=cv, scoring='accuracy')\n",
            "        threshold_results.append({\n",
            "            'threshold': thresh,\n",
            "            'model': model_name,\n",
            "            'f1_mean': f1_scores.mean(),\n",
            "            'acc_mean': acc_scores.mean(),\n",
            "        })\n",
            "        print(f'{thresh:>10.1f} | {model_name:>25s} | {f1_scores.mean():>10.4f} | {acc_scores.mean():>10.4f}')\n",
            "\n",
            "threshold_results_df = pd.DataFrame(threshold_results)\n",
            "print('\\nThreshold testing complete!')"
        ]
    },
    # --- Markdown: Results Analysis ---
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Threshold Analysis Results & Majority Voting"
        ]
    },
    # --- Code: Best threshold per model + majority vote ---
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Best Threshold Per Model ---\n",
            "print('=' * 70)\n",
            "print('  BEST THRESHOLD PER MODEL (by F1-Score)')\n",
            "print('=' * 70)\n",
            "\n",
            "best_per_model = {}\n",
            "for model_name in test_models.keys():\n",
            "    model_data = threshold_results_df[threshold_results_df['model'] == model_name]\n",
            "    best_idx = model_data['f1_mean'].idxmax()\n",
            "    best_row = model_data.loc[best_idx]\n",
            "    best_per_model[model_name] = best_row['threshold']\n",
            "    print(f'  {model_name:25s} -> Best Threshold = {best_row[\"threshold\"]:.1f}  (F1={best_row[\"f1_mean\"]:.4f})')\n",
            "\n",
            "# --- Majority Voting ---\n",
            "from collections import Counter\n",
            "votes = Counter(best_per_model.values())\n",
            "print(f'\\n  MAJORITY VOTING RESULTS:')\n",
            "for thresh, count in votes.most_common():\n",
            "    model_list = [m for m, t in best_per_model.items() if t == thresh]\n",
            "    print(f'    Score >= {thresh:.1f} : {count} votes from {model_list}')\n",
            "\n",
            "winner_threshold = votes.most_common(1)[0][0]\n",
            "print(f'\\n  >>> WINNER BY MAJORITY VOTE: Score >= {winner_threshold:.1f} <<<')\n",
            "\n",
            "# --- Average F1 Across All Models ---\n",
            "avg_per_threshold = threshold_results_df.groupby('threshold')['f1_mean'].mean().reset_index()\n",
            "avg_per_threshold.columns = ['Threshold', 'Avg_F1']\n",
            "print(f'\\n  AVERAGE F1 ACROSS ALL MODELS:')\n",
            "for _, row in avg_per_threshold.iterrows():\n",
            "    marker = ' <<<' if row['Threshold'] == avg_per_threshold.loc[avg_per_threshold['Avg_F1'].idxmax(), 'Threshold'] else ''\n",
            "    print(f'    Score >= {row[\"Threshold\"]:.1f}  =>  Avg F1 = {row[\"Avg_F1\"]:.4f}{marker}')\n",
            "\n",
            "best_avg_thresh = avg_per_threshold.loc[avg_per_threshold['Avg_F1'].idxmax(), 'Threshold']\n",
            "print(f'\\n  >>> BEST BY AVERAGE F1: Score >= {best_avg_thresh:.1f} <<<')\n",
            "print('=' * 70)"
        ]
    },
    # --- Code: Plot the results ---
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Visualization ---\n",
            "fig, axes = plt.subplots(1, 2, figsize=(18, 7))\n",
            "\n",
            "# Left: F1 vs Threshold for each model\n",
            "colors = ['#2196F3', '#4CAF50', '#9C27B0', '#F44336', '#00BCD4', '#795548', '#E91E63']\n",
            "for i, model_name in enumerate(test_models.keys()):\n",
            "    model_data = threshold_results_df[threshold_results_df['model'] == model_name]\n",
            "    axes[0].plot(model_data['threshold'], model_data['f1_mean'], 'o-',\n",
            "                 color=colors[i], linewidth=2, markersize=8, label=model_name)\n",
            "\n",
            "axes[0].axvline(x=winner_threshold, color='black', linestyle='--', linewidth=2,\n",
            "                label=f'Majority Vote = {winner_threshold:.1f}')\n",
            "axes[0].set_xlabel('Score Threshold', fontsize=12)\n",
            "axes[0].set_ylabel('F1-Score (5-Fold CV)', fontsize=12)\n",
            "axes[0].set_title('F1-Score vs Score Threshold (All Models)', fontsize=14, fontweight='bold')\n",
            "axes[0].legend(fontsize=9, loc='best')\n",
            "axes[0].grid(alpha=0.3)\n",
            "\n",
            "# Right: Average F1 bar chart\n",
            "bar_colors = ['#FF9800' if t == winner_threshold else '#2196F3' for t in avg_per_threshold['Threshold']]\n",
            "axes[1].bar(avg_per_threshold['Threshold'].astype(str), avg_per_threshold['Avg_F1'],\n",
            "            color=bar_colors, edgecolor='white', alpha=0.9)\n",
            "for i, row in avg_per_threshold.iterrows():\n",
            "    axes[1].text(i, row['Avg_F1'] + 0.001, f'{row[\"Avg_F1\"]:.4f}',\n",
            "                 ha='center', fontsize=10, fontweight='bold')\n",
            "axes[1].set_xlabel('Score Threshold', fontsize=12)\n",
            "axes[1].set_ylabel('Average F1-Score (All Models)', fontsize=12)\n",
            "axes[1].set_title('Average F1 Across All Models (Orange = Winner)', fontsize=14, fontweight='bold')\n",
            "axes[1].grid(axis='y', alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(f'\\nCONCLUSION: Using Score >= {winner_threshold:.1f} as the classification threshold.')\n",
            "print(f'This threshold was selected via majority voting across {len(test_models)} models.')\n",
            "print(f'It represents the Score value where the most models achieve their highest F1-Score.')"
        ]
    },
    # --- Updated markdown for label creation ---
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Create Target Label and Preprocess\n",
            "\n",
            "Based on the threshold analysis above, we use the majority-vote winner as our classification threshold."
        ]
    },
]

# Insert the threshold analysis cells BEFORE the old "3. Create Target Label" markdown
for j, cell in enumerate(threshold_cells):
    cells.insert(insert_idx + j, cell)

# Update the OLD "3. Create Target Label" markdown (now shifted)
old_label_md_idx = insert_idx + len(threshold_cells)
cells[old_label_md_idx]['source'] = ["## (Previous section header - can be deleted)"]

# Renumber all subsequent section headers
# The old sections were: 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
# New sections: 3=threshold analysis, 4=label creation, 5=split, 6=eval helper,
# 7=RF, 8=SVM, 9=LR, 10=NN, 11=CNN, 12=AdaBoost, 13=GB, 14=XGB, 15=Ensemble,
# 16=Comparison, 17=Confusion, 18=Bar, 19=Final
renumber = {
    '## 4. Split 80/20': '## 5. Split 80/20',
    '## 5. Evaluation Helper': '## 6. Evaluation Helper',
    '## 6. Random Forest': '## 7. Random Forest',
    '## 7. SVM': '## 8. SVM',
    '## 8. Logistic Regression': '## 9. Logistic Regression',
    '## 9. Neural Network (MLP)': '## 10. Neural Network (MLP)',
    '## 10. 1D CNN': '## 11. 1D CNN',
    '## 11. AdaBoost': '## 12. AdaBoost',
    '## 12. Gradient Boosting': '## 13. Gradient Boosting',
    '## 13. XGBoost': '## 14. XGBoost',
    '## 14. Ensemble (Majority Voting)': '## 15. Ensemble (Majority Voting)',
    '## 15. Model Comparison Table': '## 16. Model Comparison Table',
    '## 16. Confusion Matrices': '## 17. Confusion Matrices',
    '## 17. Bar Chart': '## 18. Bar Chart',
    '## 18. Final': '## 19. Final',
}

for c in cells:
    if c['cell_type'] == 'markdown' and c['source']:
        src = ''.join(c['source']).strip()
        for old, new in renumber.items():
            if src == old:
                c['source'] = [new]
                break

# Save
with open('main_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done! Notebook now has {len(cells)} cells")
print("Added: Score distribution, threshold testing with all 7 models, majority voting, and plots")
