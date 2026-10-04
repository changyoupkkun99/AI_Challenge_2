from pathlib import Path
import csv, hashlib, json
P = Path(__file__).resolve().parent
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):
    return json.loads(path.read_text(encoding="utf-8"))
copy_manifest = load(P / "copied_file_manifest.json")
missing_before_rename = set(copy_manifest.get("missing_before_rename", []))
records = [r for r in copy_manifest["files"] if r["packaged_path"] not in missing_before_rename]
for rec in records:
    path = P / rec["packaged_path"]
    assert path.is_file(), f"Missing file: {path}"
    assert digest(path) == rec.get("current_packaged_sha256", rec["sha256"]), f"Changed file: {path}"
manifest = load(P / "02_input_snapshots/relative_input_manifest.json")
plan = load(P / "02_input_snapshots/relative_delivery_plan.json")
ids = [f"test_{i:04d}.jpg" for i in range(1, 6715)]
def answers(path):
    with path.open(encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        assert reader.fieldnames == ["id", "answer"], path
        rows = list(reader)
    assert [r["id"] for r in rows] == ids, path
    assert all(r["answer"] in {"a", "b", "c", "d"} for r in rows), path
    return [r["answer"] for r in rows]
for rec in manifest["consensus_models"] + manifest["auxiliary"] + manifest["score_records"]:
    path = P / rec["path"]
    assert digest(path) == rec["sha256"], path
    answers(path)
base = answers(P / plan["baseline"]["path"])
models = [answers(P / rec["path"]) for rec in plan["consensus_models"]]
final_path = P / "01_rf_v1_original_run/RF_v1.csv"
final = answers(final_path)
locked = [all(m[i] == models[0][i] for m in models) for i in range(len(ids))]
assert sum(locked) == 4444
assert all(final[i] == models[0][i] for i, lock in enumerate(locked) if lock)
changes = [{"id": ids[i], "before": a, "after": b} for i, (a, b) in enumerate(zip(base, final)) if a != b]
expected = [{k: r[k] for k in ["id", "before", "after"]} for r in plan["changes"]]
assert changes == expected and len(changes) == 12
assert all(not locked[ids.index(r["id"])] for r in changes)
assert final[46] == "d"
# Reconstruct intended bytes in memory only; no CSV is created or modified.
intended = base.copy()
for rec in expected:
    i = ids.index(rec["id"])
    assert intended[i] == rec["before"]
    intended[i] = rec["after"]
expected_bytes = ("id,answer\n" + "".join(f"{i},{a}\n" for i,a in zip(ids,intended))).encode()
assert expected_bytes == final_path.read_bytes()
score = load(P / "01_rf_v1_original_run/RF_v1_score.json")
assert digest(final_path) == score["sha256"] and score["score"] == "0.98242"
assert len(manifest["unique_score_records"]) == 60
assert all(r["sha256"] != score["sha256"] for r in manifest["score_records"])
print(json.dumps({"copied_files_verified": len(records), "missing_before_rename": len(missing_before_rename), "rows":len(ids), "consensus_preserved":sum(locked), "changes":len(changes), "saved_change_plan_matches_final_bytes":True, "historical_scored_submissions":60, "RF_v1_score_excluded_from_training_snapshot":True, "RF_v1_score":"0.98242", "RF_v1_sha256":digest(final_path)}, ensure_ascii=False, indent=2))
