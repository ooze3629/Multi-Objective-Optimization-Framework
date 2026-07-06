import glob
import pickle
import shutil
from pathlib import Path

TARGET_SEEDS = list(range(1, 31))
OLD_SEEDS = list(range(1, 11))

for path in glob.glob("formal_results*.pkl"):
    p = Path(path)

    # 先備份原本 10 seed 結果
    backup = p.with_suffix(p.suffix + ".bak10")
    if not backup.exists():
        shutil.copy2(p, backup)

    with open(p, "rb") as f:
        data = pickle.load(f)

    meta = data.setdefault("meta", {})
    old_seeds = meta.get("seeds")

    print(f"{p.name}: old seeds = {old_seeds}")

    if old_seeds == OLD_SEEDS:
        meta["seeds"] = TARGET_SEEDS
        with open(p, "wb") as f:
            pickle.dump(data, f)
        print("  -> updated to seeds 1~30")
    elif old_seeds == TARGET_SEEDS:
        print("  -> already 1~30, skipped")
    else:
        print("  -> skipped, seeds format not expected")