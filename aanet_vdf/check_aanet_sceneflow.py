#!/usr/bin/env python3
"""
check_aanet_sceneflow.py — verify AANet's SceneFlow filename lists resolve
against the SceneFlow_hf tree, BEFORE starting a 64-epoch AANet run.

AANet (haofeixu/aanet) uses `filenames/SceneFlow_finalpass_{train,test,val}.txt`
whose paths look like:
    FlyingThings3D/frames_finalpass/TEST/A/0000/left/0006.png
    Monkaa/frames_finalpass/lonetree_augmented1_x2/left/0195.png
joined to --data_dir. This MATCHES SceneFlow_hf's native layout
({FlyingThings3D,Monkaa,Driving}/frames_finalpass/...), so NO symlink view is
needed — but we validate by sampling each list and checking the files exist.

Usage (run inside the AANet repo so filenames/ resolves):
    cd $WORKSPACE/aanet_base && python $VDFNET_ROOT/aanet_vdf/check_aanet_sceneflow.py
    # optional: pass a different data root
    python check_aanet_sceneflow.py $VDFNET_DATA
"""
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VDFNET_DATA", "data/SceneFlow")
LISTS = [
    "SceneFlow_finalpass_train.txt",
    "SceneFlow_finalpass_test.txt",
    "SceneFlow_finalpass_val.txt",
]


def main():
    all_ok = True
    print(f"[check] data root: {ROOT}")
    for name in LISTS:
        path = os.path.join("filenames", name)
        if not os.path.isfile(path):
            print(f"   {name}: LIST MISSING at {path}")
            all_ok = False
            continue
        lines = open(path).read().splitlines()
        step = max(1, len(lines) // 300)
        sample = lines[::step]
        miss = 0
        first_miss = None
        prefixes = {}
        for ln in sample:
            parts = ln.split()
            if len(parts) < 3:
                continue
            pre = parts[0].split("/")[0]
            prefixes[pre] = prefixes.get(pre, 0) + 1
            for x in parts[:3]:
                if not os.path.exists(os.path.join(ROOT, x)):
                    miss += 1
                    if first_miss is None:
                        first_miss = x
                    break
        status = "OK" if miss == 0 else f"MISSING {miss}"
        print(f"   {name}: total={len(lines)} sampled={len(sample)} "
              f"[{status}] prefixes={prefixes}"
              + (f" first_miss={first_miss}" if first_miss else ""))
        if miss:
            all_ok = False
    print("-" * 56)
    if all_ok:
        print(f"[PASS] AANet lists resolve. Train with:")
        print(f"       --data_dir {ROOT}   (NO symlink view needed)")
    else:
        print("[WARN] some paths unresolved — paste this back before training.")


if __name__ == "__main__":
    main()
