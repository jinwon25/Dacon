import json, sys
from pathlib import Path

nb_path = Path(__file__).resolve().parents[1] / 'notebooks' / 'reference.ipynb'
nb = json.loads(nb_path.read_text(encoding='utf-8'))
cells = nb['cells']
print(f'total cells: {len(cells)}')
for i, c in enumerate(cells):
    src = ''.join(c.get('source', []))
    print(f'--- cell {i} ({c["cell_type"]}, {len(src)} chars) ---')
    print(src[:600])
    print()
