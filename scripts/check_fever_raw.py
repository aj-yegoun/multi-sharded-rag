from pathlib import Path
import json
import csv

base = Path("data/raw/beir/fever")

print("Files:")
for p in base.rglob("*"):
    if p.is_file():
        print(p)

queries_path = base / "queries.jsonl"
if queries_path.exists():
    q_count = sum(1 for _ in queries_path.open("r", encoding="utf-8"))
    print("\nqueries.jsonl count =", q_count)
else:
    print("[WARN] queries.jsonl not found")

print("\nQrels:")
qrels_dir = base / "qrels"
for p in sorted(qrels_dir.glob("*.tsv")):
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    qids = {str(r["query-id"]) for r in rows}
    pos_qids = {str(r["query-id"]) for r in rows if int(r["score"]) > 0}
    pos_rows = [r for r in rows if int(r["score"]) > 0]
    pos_docs = {str(r["corpus-id"]) for r in pos_rows}

    print(f"\n{p.name}")
    print("  qrel rows          :", len(rows))
    print("  unique qids        :", len(qids))
    print("  positive qids      :", len(pos_qids))
    print("  positive rows      :", len(pos_rows))
    print("  unique positive doc:", len(pos_docs))
    print("  first 5:")
    for r in rows[:5]:
        print(" ", r)