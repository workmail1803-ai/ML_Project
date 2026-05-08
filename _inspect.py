import json
nb = json.load(open('main_model.ipynb', 'r', encoding='utf-8'))
print(f'Total cells: {len(nb["cells"])}')
for i, c in enumerate(nb['cells']):
    first_line = c['source'][0][:90].strip() if c['source'] else '(empty)'
    print(f'Cell {i}: {c["cell_type"]:8s} | {first_line}')
