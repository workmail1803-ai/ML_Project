import json

with open('main_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the threshold testing code cell (Cell 8 - the one with cross_val_score and test_models)
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and c['source']:
        src = ''.join(c['source'])
        if 'cross_val_score' in src and 'test_models' in src and 'candidate_thresholds' in src:
            print(f"Found threshold testing cell at index {i}")
            print(f"First line: {c['source'][0][:80]}")
            
            # Update this cell to include CNN and Ensemble
            c['source'] = [
                "from sklearn.model_selection import cross_val_score, StratifiedKFold\n",
                "from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.neural_network import MLPClassifier\n",
                "from xgboost import XGBClassifier\n",
                "from sklearn.base import clone\n",
                "from sklearn.metrics import f1_score as f1_metric, accuracy_score as acc_metric\n",
                "from scipy.stats import mode as scipy_mode\n",
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
                "# Define all sklearn models\n",
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
                "# CNN builder function\n",
                "def build_cnn_thresh(input_shape):\n",
                "    m = Sequential([\n",
                "        Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),\n",
                "        BatchNormalization(), MaxPooling1D(2), Dropout(0.3),\n",
                "        Conv1D(128, 3, activation='relu', padding='same'),\n",
                "        BatchNormalization(), MaxPooling1D(2), Dropout(0.3),\n",
                "        Conv1D(64, 3, activation='relu', padding='same'),\n",
                "        BatchNormalization(), Flatten(),\n",
                "        Dense(128, activation='relu'), Dropout(0.5),\n",
                "        Dense(64, activation='relu'), Dropout(0.3),\n",
                "        Dense(1, activation='sigmoid')\n",
                "    ])\n",
                "    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n",
                "    return m\n",
                "\n",
                "candidate_thresholds = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0]\n",
                "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
                "threshold_results = []\n",
                "\n",
                "print(f'Testing {len(candidate_thresholds)} thresholds x {len(test_models) + 2} models (incl. 1D CNN & Ensemble)...')\n",
                "print(f'{\"Threshold\":>10s} | {\"Model\":>25s} | {\"F1-Score\":>10s} | {\"Accuracy\":>10s}')\n",
                "print('-' * 65)\n",
                "\n",
                "for thresh in candidate_thresholds:\n",
                "    y_temp = (df_thresh_test['Score'] >= thresh).astype(int)\n",
                "    X_temp = df_thresh_test.drop(columns=['Score'])\n",
                "    scaler_temp = StandardScaler()\n",
                "    X_temp_scaled = scaler_temp.fit_transform(X_temp)\n",
                "\n",
                "    # --- sklearn models ---\n",
                "    for model_name, model in test_models.items():\n",
                "        f1_scores = cross_val_score(model, X_temp_scaled, y_temp, cv=cv, scoring='f1')\n",
                "        acc_scores = cross_val_score(model, X_temp_scaled, y_temp, cv=cv, scoring='accuracy')\n",
                "        threshold_results.append({\n",
                "            'threshold': thresh, 'model': model_name,\n",
                "            'f1_mean': f1_scores.mean(), 'acc_mean': acc_scores.mean(),\n",
                "        })\n",
                "        print(f'{thresh:>10.1f} | {model_name:>25s} | {f1_scores.mean():>10.4f} | {acc_scores.mean():>10.4f}')\n",
                "\n",
                "    # --- 1D CNN (manual CV) ---\n",
                "    cnn_f1, cnn_acc = [], []\n",
                "    for train_idx, test_idx in cv.split(X_temp_scaled, y_temp):\n",
                "        X_tr, X_te = X_temp_scaled[train_idx], X_temp_scaled[test_idx]\n",
                "        y_tr, y_te = y_temp.values[train_idx], y_temp.values[test_idx]\n",
                "        X_tr_c = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)\n",
                "        X_te_c = X_te.reshape(X_te.shape[0], X_te.shape[1], 1)\n",
                "        cnn_m = build_cnn_thresh((X_tr.shape[1], 1))\n",
                "        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)\n",
                "        cnn_m.fit(X_tr_c, y_tr, epochs=50, batch_size=32, validation_split=0.2, callbacks=[es], verbose=0)\n",
                "        pred = (cnn_m.predict(X_te_c, verbose=0).flatten() >= 0.5).astype(int)\n",
                "        cnn_f1.append(f1_metric(y_te, pred))\n",
                "        cnn_acc.append(acc_metric(y_te, pred))\n",
                "    threshold_results.append({'threshold': thresh, 'model': '1D CNN',\n",
                "        'f1_mean': np.mean(cnn_f1), 'acc_mean': np.mean(cnn_acc)})\n",
                "    print(f'{thresh:>10.1f} | {\"1D CNN\":>25s} | {np.mean(cnn_f1):>10.4f} | {np.mean(cnn_acc):>10.4f}')\n",
                "\n",
                "    # --- Ensemble (majority vote of all 8 models per fold) ---\n",
                "    ens_f1, ens_acc = [], []\n",
                "    for train_idx, test_idx in cv.split(X_temp_scaled, y_temp):\n",
                "        X_tr, X_te = X_temp_scaled[train_idx], X_temp_scaled[test_idx]\n",
                "        y_tr, y_te = y_temp.values[train_idx], y_temp.values[test_idx]\n",
                "        preds = []\n",
                "        for m_obj in test_models.values():\n",
                "            mc = clone(m_obj)\n",
                "            mc.fit(X_tr, y_tr)\n",
                "            preds.append(mc.predict(X_te))\n",
                "        X_tr_c = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)\n",
                "        X_te_c = X_te.reshape(X_te.shape[0], X_te.shape[1], 1)\n",
                "        cnn_m = build_cnn_thresh((X_tr.shape[1], 1))\n",
                "        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)\n",
                "        cnn_m.fit(X_tr_c, y_tr, epochs=50, batch_size=32, validation_split=0.2, callbacks=[es], verbose=0)\n",
                "        preds.append((cnn_m.predict(X_te_c, verbose=0).flatten() >= 0.5).astype(int))\n",
                "        ens_p, _ = scipy_mode(np.array(preds), axis=0, keepdims=False)\n",
                "        ens_p = ens_p.flatten()\n",
                "        ens_f1.append(f1_metric(y_te, ens_p))\n",
                "        ens_acc.append(acc_metric(y_te, ens_p))\n",
                "    threshold_results.append({'threshold': thresh, 'model': 'Ensemble',\n",
                "        'f1_mean': np.mean(ens_f1), 'acc_mean': np.mean(ens_acc)})\n",
                "    print(f'{thresh:>10.1f} | {\"Ensemble\":>25s} | {np.mean(ens_f1):>10.4f} | {np.mean(ens_acc):>10.4f}')\n",
                "\n",
                "threshold_results_df = pd.DataFrame(threshold_results)\n",
                "print('\\nThreshold testing complete!')"
            ]
            c['outputs'] = []
            print("  -> Updated with CNN + Ensemble")
            break

# Now find and update the majority voting results cell
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and c['source']:
        src = ''.join(c['source'])
        if 'best_per_model' in src and 'MAJORITY VOTING' in src:
            print(f"Found majority voting cell at index {i}")
            # Update to include CNN and Ensemble model names
            c['source'] = [
                "# --- Best Threshold Per Model ---\n",
                "print('=' * 70)\n",
                "print('  BEST THRESHOLD PER MODEL (by F1-Score)')\n",
                "print('=' * 70)\n",
                "\n",
                "all_tested_models = list(test_models.keys()) + ['1D CNN', 'Ensemble']\n",
                "best_per_model = {}\n",
                "for model_name in all_tested_models:\n",
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
                "# --- Average F1 Across All 9 Models ---\n",
                "avg_per_threshold = threshold_results_df.groupby('threshold')['f1_mean'].mean().reset_index()\n",
                "avg_per_threshold.columns = ['Threshold', 'Avg_F1']\n",
                "print(f'\\n  AVERAGE F1 ACROSS ALL 9 MODELS:')\n",
                "for _, row in avg_per_threshold.iterrows():\n",
                "    marker = ' <<<' if row['Threshold'] == avg_per_threshold.loc[avg_per_threshold['Avg_F1'].idxmax(), 'Threshold'] else ''\n",
                "    print(f'    Score >= {row[\"Threshold\"]:.1f}  =>  Avg F1 = {row[\"Avg_F1\"]:.4f}{marker}')\n",
                "\n",
                "best_avg_thresh = avg_per_threshold.loc[avg_per_threshold['Avg_F1'].idxmax(), 'Threshold']\n",
                "print(f'\\n  >>> BEST BY AVERAGE F1: Score >= {best_avg_thresh:.1f} <<<')\n",
                "print('=' * 70)"
            ]
            c['outputs'] = []
            print("  -> Updated majority voting cell")
            break

# Update the visualization cell to show all 9 models
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and c['source']:
        src = ''.join(c['source'])
        if 'Visualization' in src and 'winner_threshold' in src and 'axes[0].plot' in src:
            print(f"Found visualization cell at index {i}")
            c['source'] = [
                "# --- Visualization ---\n",
                "fig, axes = plt.subplots(1, 2, figsize=(18, 7))\n",
                "\n",
                "# Left: F1 vs Threshold for each model\n",
                "colors = ['#2196F3', '#4CAF50', '#9C27B0', '#F44336', '#00BCD4',\n",
                "          '#795548', '#E91E63', '#FF9800', '#607D8B']\n",
                "all_tested_models = list(test_models.keys()) + ['1D CNN', 'Ensemble']\n",
                "for i, model_name in enumerate(all_tested_models):\n",
                "    model_data = threshold_results_df[threshold_results_df['model'] == model_name]\n",
                "    axes[0].plot(model_data['threshold'], model_data['f1_mean'], 'o-',\n",
                "                 color=colors[i], linewidth=2, markersize=8, label=model_name)\n",
                "\n",
                "axes[0].axvline(x=winner_threshold, color='black', linestyle='--', linewidth=2,\n",
                "                label=f'Majority Vote = {winner_threshold:.1f}')\n",
                "axes[0].set_xlabel('Score Threshold', fontsize=12)\n",
                "axes[0].set_ylabel('F1-Score (5-Fold CV)', fontsize=12)\n",
                "axes[0].set_title('F1-Score vs Score Threshold (All 9 Models)', fontsize=14, fontweight='bold')\n",
                "axes[0].legend(fontsize=8, loc='best')\n",
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
                "axes[1].set_ylabel('Average F1-Score (All 9 Models)', fontsize=12)\n",
                "axes[1].set_title('Average F1 Across All 9 Models (Orange = Winner)', fontsize=14, fontweight='bold')\n",
                "axes[1].grid(axis='y', alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print(f'\\nCONCLUSION: Using Score >= {winner_threshold:.1f} as the classification threshold.')\n",
                "print(f'This threshold was selected via majority voting across all 9 models (7 sklearn + 1D CNN + Ensemble).')\n",
                "print(f'It represents the Score value where the most models achieve their highest F1-Score.')"
            ]
            c['outputs'] = []
            print("  -> Updated visualization cell")
            break

with open('main_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nDone! All threshold cells now include CNN + Ensemble.")
