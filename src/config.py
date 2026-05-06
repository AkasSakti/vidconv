import os
from pathlib import Path


COLAB_ROOT = Path("/content/drive/My Drive/Colab Notebooks/vidconv")
LOCAL_ROOT = Path(__file__).resolve().parents[1]


def resolve_root_dir() -> Path:
    """Prefer the Google Drive project path in Colab, otherwise use this repo."""
    env_root = os.getenv("VIDCONV_ROOT")
    if env_root:
        return Path(env_root)
    if COLAB_ROOT.exists():
        return COLAB_ROOT
    return LOCAL_ROOT


ROOT_DIR = resolve_root_dir()
RAW_DATASET_DIR = ROOT_DIR / "dataset"
GENERATED_DATASET_DIR = ROOT_DIR / "generated_dataset"
RUNS_DIR = ROOT_DIR / "runs"
MODELS_DIR = ROOT_DIR / "models"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_NAMES = [
    "Good_Day",
    "Kapal_Api",
    "Kopi_Gadjah",
    "Luwak_White_Coffe",
    "Nutrisari",
    "Torabika",
]


def discover_classes(dataset_dir: Path = RAW_DATASET_DIR) -> list[str]:
    """Return class names from subdirectories that contain at least one image."""
    if not dataset_dir.exists():
        return []

    classes: list[str] = []
    for folder in sorted(dataset_dir.iterdir()):
        if not folder.is_dir():
            continue
        has_image = any(path.suffix.lower() in IMAGE_EXTENSIONS for path in folder.iterdir() if path.is_file())
        if has_image:
            classes.append(folder.name)
    return classes


CLASS_NAMES = discover_classes() or DEFAULT_CLASS_NAMES
