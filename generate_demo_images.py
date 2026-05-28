#!/usr/bin/env python3
"""
generate_demo_images.py — Pre-generate images for the SAEmnesia interactive demo.

Calls the live inference server at https://hssh1.di.unito.it:5443/generate and
saves the results in a directory tree that the static GitHub page can reference
directly without running any server-side inference.

Output layout
─────────────
demo_images/
  {object}/{style}/
    base.png                        ← "without_sae" (saved once per object/style)
    {unlearn}/t{ts}/
      optimal/                      ← SAEuron at its optimal mult, SAEmnesia at its own
        saeuron.png
        saemnesia.png
      m5/                           ← both methods at −5
        saeuron.png
        saemnesia.png
      m10/ m20/ m30/  (same pattern)
  manifest.json
  manifest.js                       ← for the HTML page

Usage
─────
  python generate_demo_images.py              # generate everything
  python generate_demo_images.py --dry-run    # print combos, don't call API
  python generate_demo_images.py --delay 2    # 2 s gap between calls
  python generate_demo_images.py --output-dir /path/to/dir
"""

import base64
import json
import time
import argparse
import urllib3
from pathlib import Path
from itertools import product

import requests
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ──────────────────────────────────────────────────── Configuration ───────────

API_URL = "https://hssh1.di.unito.it:5443/generate"

ALL_OBJECTS = [
    "Architectures", "Bears", "Birds", "Butterfly", "Cats",
    "Dogs", "Fishes", "Flame", "Flowers", "Frogs",
    "Horses", "Human", "Jellyfish", "Rabbits", "Sandwiches",
    "Sea", "Statues", "Towers", "Trees", "Waterfalls",
]

# 5 fixed companion objects always generated alongside the unlearn target
COMPANION_OBJECTS = ["Dogs", "Cats", "Birds", "Flowers", "Horses"]

STYLES = [
    "Cartoon",
    "Watercolor",
    "Van_Gogh",
]

# "None" = no concept suppression (plain SAE reconstruction at mult=0)
UNLEARN_TARGETS = ["None"] + ALL_OBJECTS

# start_timestep is 0-based (0–steps-1). None = full run (leave blank in UI).
TIMESTEPS = [None, 25]

# Shared multipliers (in addition to optimal)
SHARED_MULTIPLIERS = [-1.0, -5.0, -10.0]

# Per-concept optimal multipliers (from the demo app frontend)
OPTIMAL_SAEMNESIA = {
    "Architectures": -5.0,  "Bears": -20.0, "Birds": -5.0,  "Butterfly": -5.0,
    "Cats": -15.0,           "Dogs": -5.0,   "Fishes": -5.0, "Flame": -1.0,
    "Flowers": -5.0,         "Frogs": -15.0, "Horses": -20.0,"Human": -5.0,
    "Jellyfish": -1.0,       "Rabbits": -5.0,"Sandwiches": -15.0,"Sea": -5.0,
    "Statues": -20.0,        "Towers": -5.0, "Trees": -5.0,  "Waterfalls": -5.0,
}

OPTIMAL_SAEURON = {
    "Architectures": -30.0, "Bears": -5.0,  "Birds": -5.0,  "Butterfly": -5.0,
    "Cats": -30.0,           "Dogs": -15.0,  "Fishes": -30.0,"Flame": -30.0,
    "Flowers": -20.0,        "Frogs": -30.0, "Horses": -20.0,"Human": -5.0,
    "Jellyfish": -1.0,       "Rabbits": -10.0,"Sandwiches": -10.0,"Sea": -5.0,
    "Statues": -5.0,         "Towers": -1.0, "Trees": -20.0, "Waterfalls": -1.0,
}

SEED = 42
STEPS = 50
GUIDANCE = 7.5
HOOKPOINT = "unet_up_blocks_1_attentions_1"

# ────────────────────────────────────────────────────────── Helpers ───────────

def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def save_image(data_url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = data_url.split(",", 1)[-1] if "," in data_url else data_url
    path.write_bytes(base64.b64decode(payload))


def call_api(
    prompt: str,
    unlearn: str,
    mult_saemnesia: float,
    mult_saeuron: float,
    start_ts: int,
) -> dict:
    payload = {
        "prompt": prompt,
        "object_to_unlearn": "" if unlearn == "None" else unlearn,
        "multiplier_saemnesia": mult_saemnesia,
        "multiplier_saeuron": mult_saeuron,
        "multiplier_saemnesia_btk": 0.0,
        "padding": 0,
        "seed": SEED,
        "steps": STEPS,
        "guidance_scale": GUIDANCE,
        "hookpoint": HOOKPOINT,
        "start_timestep": start_ts,
        "end_timestep": None,
    }
    r = requests.post(API_URL, json=payload, timeout=300, verify=False)
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(f"API error: {data.get('error', 'unknown')}")
    return data["images"]


# ─────────────────────────────────────────────────── Combo builders ───────────

def objects_for_unlearn(unlearn: str) -> list[str]:
    """
    Return the objects to generate when a given concept is being unlearned.
    Always includes the unlearn target itself + the 5 fixed companion objects,
    deduplicated (so targets already in COMPANION_OBJECTS yield 5 total).
    For "None", just use the companions.
    """
    if unlearn == "None":
        return list(COMPANION_OBJECTS)
    return list(dict.fromkeys([unlearn] + COMPANION_OBJECTS))  # ordered, deduped


def build_combos(styles, unlearn_targets, timesteps):
    """
    Returns a list of dicts, each describing one API call and where to save results.

    Two kinds of entries:
      kind="optimal"  — SAEuron and SAEmnesia each use their own per-concept
                        optimal multiplier; results go into .../optimal/
      kind="shared"   — both methods use the same user-chosen multiplier;
                        results go into .../m{X}/
    """
    combos = []

    for unlearn, style, ts in product(unlearn_targets, styles, timesteps):
        for obj in objects_for_unlearn(unlearn):
            prompt = f"An image of {obj} in {style} style."
            base_path = Path("demo_images") / slug(obj) / slug(style) / "base.png"
            ts_tag = "tfull" if ts is None else f"t{ts}"
            ts_dir = Path("demo_images") / slug(obj) / slug(style) / slug(unlearn) / ts_tag

            m_saemn   = 0.0 if unlearn == "None" else OPTIMAL_SAEMNESIA.get(unlearn, -5.0)
            m_saeuron = 0.0 if unlearn == "None" else OPTIMAL_SAEURON.get(unlearn, -5.0)

            combos.append(dict(
                kind="optimal",
                obj=obj, style=style, unlearn=unlearn, ts=ts,
                prompt=prompt,
                mult_saemnesia=m_saemn,
                mult_saeuron=m_saeuron,
                base_path=base_path,
                saeuron_path=ts_dir / "optimal" / "saeuron.png",
                saemnesia_path=ts_dir / "optimal" / "saemnesia.png",
            ))

            # Shared-multiplier combos (skip when unlearn=None)
            if unlearn != "None":
                for mult in SHARED_MULTIPLIERS:
                    tag = f"m{abs(int(mult))}"
                    combos.append(dict(
                        kind="shared",
                        obj=obj, style=style, unlearn=unlearn, ts=ts,
                        prompt=prompt,
                        mult_saemnesia=mult,
                        mult_saeuron=mult,
                        base_path=base_path,
                        saeuron_path=ts_dir / tag / "saeuron.png",
                        saemnesia_path=ts_dir / tag / "saemnesia.png",
                    ))

    return combos


# ─────────────────────────────────────────────────────────────── Main ────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    combos = build_combos(STYLES, UNLEARN_TARGETS, TIMESTEPS)

    total = len(combos)
    opt_count    = sum(1 for c in combos if c["kind"] == "optimal")
    shared_count = total - opt_count
    print(f"Total combos: {total:,}  ({opt_count} optimal + {shared_count} shared-multiplier)")

    if args.dry_run:
        for c in combos:
            tag = "optimal" if c["kind"] == "optimal" else f"m{abs(int(c['mult_saemnesia']))}"
            ts_label = "tfull" if c["ts"] is None else f"t{c['ts']}"
            print(f"  [{c['kind']:7}] {c['obj']}/{c['style']}/{c['unlearn']}/{ts_label}/{tag}"
                  f"  saemn={c['mult_saemnesia']}  saeuron={c['mult_saeuron']}")
        return

    manifest: dict[str, bool] = {}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    errors = 0

    with tqdm(total=total, unit="img", dynamic_ncols=True) as pbar:
        for c in combos:
            saeuron_path   = out / c["saeuron_path"]
            saemnesia_path = out / c["saemnesia_path"]
            base_path      = out / c["base_path"]
            combo_key      = str(c["saeuron_path"].parent)

            tag = "optimal" if c["kind"] == "optimal" else f"m{abs(int(c['mult_saemnesia']))}"
            ts_label = "tfull" if c["ts"] is None else f"t{c['ts']}"
            pbar.set_description(f"{c['obj']}/{c['style']}/{c['unlearn']}/{ts_label}/{tag}")

            if base_path.exists() and saeuron_path.exists() and saemnesia_path.exists():
                manifest[combo_key] = True
                pbar.update(1)
                continue

            try:
                images = call_api(
                    prompt=c["prompt"],
                    unlearn=c["unlearn"],
                    mult_saemnesia=c["mult_saemnesia"],
                    mult_saeuron=c["mult_saeuron"],
                    start_ts=c["ts"],
                )

                if not base_path.exists():
                    save_image(images["without_sae"], out / c["base_path"])

                save_image(images["saeuron"],   saeuron_path)
                save_image(images["saemnesia"], saemnesia_path)

                manifest[combo_key] = True

            except Exception as exc:
                errors += 1
                tqdm.write(f"ERROR {combo_key}: {exc}")
                manifest[combo_key] = False

            manifest_path.write_text(json.dumps(manifest, indent=2))
            pbar.update(1)
            time.sleep(args.delay)

    success = sum(v for v in manifest.values())
    print(f"\nFinished: {success}/{total} succeeded, {errors} errors.")

    js_path = out / "manifest.js"
    js_path.write_text(f"const DEMO_MANIFEST = {json.dumps(manifest, indent=2)};\n")
    print(f"Manifest: {manifest_path}  |  JS: {js_path}")


if __name__ == "__main__":
    main()
