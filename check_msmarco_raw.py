from pathlib import Path
import json

base = Path("data/raw/beir/msmarco")

print("Files:")
for p in base.rglob("*"):
    if p.is_file():
        print(p)

print("\nQuery-like files:")
for p in base.rglob("*queries*"):
    print(p)

for p in base.rglob("queries.jsonl"):
    count = sum(1 for _ in p.open("r", encoding="utf-8"))
    print("queries.jsonl:", p, "count =", count)

for p in base.rglob("qrels/*.tsv"):
    lines = p.read_text(encoding="utf-8").splitlines()
    print("qrels:", p, "lines =", len(lines))
    print("first 5:")
    for line in lines[:5]:
        print(line)