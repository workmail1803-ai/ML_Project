import pandas as pd
import numpy as np

df = pd.read_csv('Malware_Analysis.csv')
print(f'Shape: {df.shape}')
print(f'\nColumns ({len(df.columns)}):')
for c in df.columns:
    print(f'  {c}: {df[c].dtype} | nulls={df[c].isnull().sum()} | unique={df[c].nunique()}')

print(f'\nScore stats:')
print(df['Score'].describe())
print(f'\nScore value counts:')
print(df['Score'].value_counts().sort_index())

print(f'\nClass balance (Score >= 6.5):')
label = (df['Score'] >= 6.5).astype(int)
print(label.value_counts())
print(f'Ratio: {label.value_counts()[1]/label.value_counts()[0]:.2f}')

print(f'\nSample rows:')
print(df.head(3).to_string())
