#!/usr/bin/env python3
"""Scan the repository for Python imports using AST and write a dependency list."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
py_files = list(ROOT.rglob('*.py'))
deps = set()

for p in py_files:
    try:
        tree = ast.parse(p.read_text(encoding='utf-8'))
    except Exception:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                deps.add(n.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                deps.add(node.module.split('.')[0])

out = ROOT / 'scripts' / 'detected_dependencies_ast.txt'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('\n'.join(sorted(deps)))
print(f"Wrote {out} with {len(deps)} detected modules")
