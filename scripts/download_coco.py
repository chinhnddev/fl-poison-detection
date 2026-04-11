from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
CHUNK_SIZE = 1024 * 1024

COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

COCO_CATEGORY_IDS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]

CAT_ID_TO_CLASS = {cid: idx for idx, cid in enumerate(COCO_CATEGORY_IDS)}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_dataset_root() -> Path:
    return repo_root() / "datasets" / "coco"


def default_yaml_path() -> Path:
    return repo_root() / "datasets" / "coco_val2017.yaml"


def _print(msg: str, *, verbose: bool = True) -> None:
    if verbose:
        print(msg)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _count_files(root: Path, suffix: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.glob(f"*{suffix}") if p.is_file())


def _download_with_resume(url: str, dest: Path, *, verbose: bool = True) -> Path:
    _ensure_dir(dest.parent)
    existing_size = dest.stat().st_size if dest.exists() else 0
    headers = {}
    mode = "wb"
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"
        _print(f"Resuming download: {dest.name} from byte {existing_size}", verbose=verbose)
    else:
        _print(f"Downloading: {url}", verbose=verbose)

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            status = getattr(resp, "status", 200)
            if existing_size > 0 and status != 206:
                _print(f"Server did not honor resume for {dest.name}; restarting download.", verbose=verbose)
                existing_size = 0
                mode = "wb"
            total_size = None
            content_range = resp.headers.get("Content-Range")
            if content_range and "/" in content_range:
                try:
                    total_size = int(content_range.rsplit("/", 1)[1])
                except ValueError:
                    total_size = None
            elif resp.headers.get("Content-Length"):
                try:
                    total_size = existing_size + int(resp.headers["Content-Length"])
                except ValueError:
                    total_size = None

            if mode == "wb" and dest.exists():
                dest.unlink()

            bytes_written = existing_size
            with open(dest, mode) as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_written += len(chunk)
                    if total_size:
                        pct = 100.0 * bytes_written / max(total_size, 1)
                        _print(
                            f"  {dest.name}: {bytes_written / (1024 * 1024):.1f} MB / "
                            f"{total_size / (1024 * 1024):.1f} MB ({pct:.1f}%)",
                            verbose=verbose,
                        )
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Failed to download {url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc.reason}") from exc

    return dest


def _extract_zip(zip_path: Path, target_dir: Path, *, verbose: bool = True) -> None:
    _ensure_dir(target_dir)
    _print(f"Extracting: {zip_path.name} -> {target_dir}", verbose=verbose)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def _iter_grouped_annotations(annotations: Iterable[Dict]) -> Dict[int, List[Dict]]:
    grouped: Dict[int, List[Dict]] = {}
    for ann in annotations:
        grouped.setdefault(int(ann["image_id"]), []).append(ann)
    return grouped


def convert_instances_to_yolo(
    annotation_json: Path,
    images_dir: Path,
    labels_dir: Path,
    *,
    verbose: bool = True,
) -> int:
    if not annotation_json.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_json}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {images_dir}")

    _ensure_dir(labels_dir)
    data = json.loads(annotation_json.read_text(encoding="utf-8"))
    grouped_annotations = _iter_grouped_annotations(data.get("annotations", []))
    written = 0

    for image in data.get("images", []):
        image_id = int(image["id"])
        width = float(image["width"])
        height = float(image["height"])
        file_name = str(image["file_name"])
        image_path = images_dir / file_name
        label_path = labels_dir / Path(file_name).with_suffix(".txt")
        label_lines: List[str] = []

        for ann in grouped_annotations.get(image_id, []):
            if int(ann.get("iscrowd", 0)) != 0:
                continue
            bbox = ann.get("bbox") or []
            if len(bbox) != 4:
                continue
            x, y, w, h = map(float, bbox)
            if w <= 0.0 or h <= 0.0 or width <= 0.0 or height <= 0.0:
                continue
            category_id = int(ann["category_id"])
            if category_id not in CAT_ID_TO_CLASS:
                continue
            xc = (x + w / 2.0) / width
            yc = (y + h / 2.0) / height
            wn = w / width
            hn = h / height
            label_lines.append(
                f"{CAT_ID_TO_CLASS[category_id]} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"
            )

        if not image_path.exists():
            raise FileNotFoundError(f"Image listed in annotations is missing: {image_path}")

        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        written += 1

    _print(f"Generated YOLO labels for {written} images in {labels_dir}", verbose=verbose)
    return written


def write_coco_yaml(yaml_path: Path, dataset_root: Path, *, verbose: bool = True) -> None:
    try:
        rel_dataset_root = dataset_root.relative_to(repo_root()).as_posix()
    except ValueError:
        rel_dataset_root = dataset_root.as_posix()
    cfg = {
        "path": rel_dataset_root,
        "train": "train.txt",
        "val": "images/val2017",
        "source_train": "images/val2017",
        "nc": len(COCO_NAMES),
        "names": COCO_NAMES,
    }
    _ensure_dir(yaml_path.parent)
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    _print(f"Wrote dataset YAML: {yaml_path}", verbose=verbose)


def is_coco_val2017_ready(dataset_root: Path) -> bool:
    images_dir = dataset_root / "images" / "val2017"
    annotation_json = dataset_root / "annotations" / "instances_val2017.json"
    labels_dir = dataset_root / "labels" / "val2017"
    return (
        images_dir.exists()
        and annotation_json.exists()
        and labels_dir.exists()
        and _count_files(images_dir, ".jpg") >= 5000
        and _count_files(labels_dir, ".txt") >= 5000
    )


def ensure_coco_val2017_dataset(
    dataset_root: Path | None = None,
    yaml_path: Path | None = None,
    *,
    keep_archives: bool = False,
    verbose: bool = True,
) -> Dict[str, str]:
    dataset_root = (dataset_root or default_dataset_root()).resolve()
    yaml_path = (yaml_path or default_yaml_path()).resolve()

    images_root = dataset_root / "images"
    images_dir = images_root / "val2017"
    annotations_dir = dataset_root / "annotations"
    annotation_json = annotations_dir / "instances_val2017.json"
    labels_dir = dataset_root / "labels" / "val2017"
    archives_dir = dataset_root / "_archives"
    val_zip = archives_dir / "val2017.zip"
    ann_zip = archives_dir / "annotations_trainval2017.zip"

    _ensure_dir(images_root)
    _ensure_dir(annotations_dir)
    _ensure_dir(labels_dir)
    _ensure_dir(archives_dir)

    image_count = _count_files(images_dir, ".jpg")
    if image_count >= 5000:
        _print(f"Images already present: {images_dir} ({image_count} jpg files)", verbose=verbose)
    else:
        _download_with_resume(VAL2017_URL, val_zip, verbose=verbose)
        _extract_zip(val_zip, images_root, verbose=verbose)

    if annotation_json.exists():
        _print(f"Annotations already present: {annotation_json}", verbose=verbose)
    else:
        _download_with_resume(ANNOTATIONS_URL, ann_zip, verbose=verbose)
        _extract_zip(ann_zip, dataset_root, verbose=verbose)

    final_image_count = _count_files(images_dir, ".jpg")
    if final_image_count < 5000:
        raise RuntimeError(f"Expected at least 5000 jpg files under {images_dir}, found {final_image_count}")
    if not annotation_json.exists():
        raise RuntimeError(f"Expected annotation file after extraction: {annotation_json}")

    label_count = _count_files(labels_dir, ".txt")
    if label_count < final_image_count:
        convert_instances_to_yolo(annotation_json, images_dir, labels_dir, verbose=verbose)
        label_count = _count_files(labels_dir, ".txt")

    if label_count < final_image_count:
        raise RuntimeError(f"Expected {final_image_count} YOLO label files under {labels_dir}, found {label_count}")

    write_coco_yaml(yaml_path, dataset_root, verbose=verbose)

    if not keep_archives:
        for archive in [val_zip, ann_zip]:
            if archive.exists():
                archive.unlink()
                _print(f"Removed archive: {archive}", verbose=verbose)

    return {
        "dataset_root": str(dataset_root),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "annotation_json": str(annotation_json),
        "yaml_path": str(yaml_path),
    }


def ensure_coco_val2017_for_yaml(data_yaml: str, *, verbose: bool = True) -> bool:
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        return False
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    source_train = str((cfg.get("source_train") or "")).strip()
    val_ref = str((cfg.get("val") or "")).strip()
    if yaml_path.name != "coco_val2017.yaml" and source_train != "images/val2017" and val_ref != "images/val2017":
        return False

    root_ref = Path(str(cfg.get("path") or "datasets/coco"))
    if root_ref.is_absolute():
        dataset_root = root_ref
    else:
        direct = (yaml_path.parent / root_ref).resolve()
        cwd_root = (Path.cwd() / root_ref).resolve()
        dataset_root = cwd_root if "datasets/coco" in str(root_ref).replace("\\", "/") else direct
        if not dataset_root.exists() and direct.exists():
            dataset_root = direct

    ensure_coco_val2017_dataset(dataset_root=dataset_root, yaml_path=yaml_path, verbose=verbose)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and prepare COCO val2017 for this repo.")
    ap.add_argument("--dataset_root", default="./datasets/coco", help="Target dataset root directory")
    ap.add_argument("--yaml", dest="yaml_path", default="./datasets/coco_val2017.yaml", help="Dataset YAML to create/update")
    ap.add_argument("--keep-archives", action="store_true", help="Keep downloaded zip archives after extraction")
    args = ap.parse_args()

    info = ensure_coco_val2017_dataset(
        dataset_root=(Path.cwd() / args.dataset_root).resolve(),
        yaml_path=(Path.cwd() / args.yaml_path).resolve(),
        keep_archives=bool(args.keep_archives),
        verbose=True,
    )
    print("COCO val2017 is ready.")
    for key, value in info.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
