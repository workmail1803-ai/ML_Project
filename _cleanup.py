import json

with open('main_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Remove the "(Previous section header - can be deleted)" cell at index 13
cells = nb['cells']
for i, c in enumerate(cells):
    if c['cell_type'] == 'markdown' and c['source']:
        if 'Previous section header' in ''.join(c['source']):
            cells.pop(i)
            print(f"Removed leftover cell at index {i}")
            break

with open('main_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Final cell count: {len(cells)}")
