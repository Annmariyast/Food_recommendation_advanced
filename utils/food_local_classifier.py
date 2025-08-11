import os
import io
import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

_tf_ok = True
_feature_model = None
_index_embeddings: Optional[np.ndarray] = None
_index_labels: List[str] = []
_index_classes: List[str] = []
_indexed_root: Optional[str] = None
_last_error: Optional[str] = None


def _ensure_model():
    global _tf_ok, _feature_model
    if _feature_model is not None or not _tf_ok:
        return
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf  # type: ignore
        from tensorflow.keras.applications import EfficientNetB0  # type: ignore
        from tensorflow.keras.applications.efficientnet import preprocess_input  # type: ignore

        base = EfficientNetB0(include_top=False, pooling="avg", weights="imagenet")
        input_shape = (224, 224)

        def embed_images(batch: np.ndarray) -> np.ndarray:
            x = preprocess_input(batch)
            return base.predict(x, verbose=0)

        _feature_model = embed_images
    except Exception:
        _tf_ok = False
        global _last_error
        _last_error = "TensorFlow/EfficientNet not available."


def _resolve_class_root(root: str) -> str:
    """Return a directory whose immediate subfolders are class folders.

    Tries depth-1 under root; if no class-like structure is found, tries
    one more level (e.g., root/images/* or root/train/*).
    """
    if not os.path.isdir(root):
        return root
    # Check if immediate children look like class directories
    children = [os.path.join(root, d) for d in os.listdir(root)]
    class_like = [d for d in children if os.path.isdir(d) and any(
        f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(d)
    )]
    if class_like:
        return root
    # Try depth-2: look for a child whose children are class dirs
    for d in children:
        if not os.path.isdir(d):
            continue
        subchildren = [os.path.join(d, s) for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
        class_like = [s for s in subchildren if any(
            f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(s)
        )]
        if len(class_like) >= 3:  # heuristic: at least a few classes
            return d
    return root


def _iter_image_paths(root: str, samples_per_class: int = 12) -> List[Tuple[str, str]]:
    """Return up to N image paths per class directory under root."""
    pairs: List[Tuple[str, str]] = []
    if not os.path.isdir(root):
        return pairs
    class_root = _resolve_class_root(root)
    for cls in sorted(os.listdir(class_root)):
        cls_dir = os.path.join(class_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not files:
            continue
        random.shuffle(files)
        for f in files[:samples_per_class]:
            pairs.append((os.path.join(cls_dir, f), cls))
    return pairs


def _load_image_to_array(path_or_bytes, size=(224, 224)) -> np.ndarray:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(path_or_bytes)).convert("RGB")
    else:
        img = Image.open(path_or_bytes).convert("RGB")
    img = img.resize(size)
    arr = np.asarray(img, dtype=np.float32)
    return arr


def build_index(dataset_root: str, samples_per_class: int = 12) -> bool:
    """Build an embedding index from a directory of class folders and cache in memory.

    dataset_root: directory containing subfolders per class.
    """
    global _index_embeddings, _index_labels, _index_classes, _indexed_root
    global _last_error
    _last_error = None
    _ensure_model()
    if not _tf_ok or _feature_model is None:
        _last_error = _last_error or "TensorFlow backend unavailable."
        return False
    pairs = _iter_image_paths(dataset_root, samples_per_class=samples_per_class)
    if not pairs:
        _last_error = (
            "No class folders with images found under the provided dataset root."
        )
        return False
    images = []
    labels = []
    classes = sorted({cls for _, cls in pairs})
    for path, cls in pairs:
        try:
            images.append(_load_image_to_array(path))
            labels.append(cls)
        except Exception as e:
            # skip unreadable image
            continue
    if not images:
        _last_error = "Failed to load any images from dataset."
        return False
    batch = np.stack(images, axis=0)
    feats = _feature_model(batch)  # type: ignore[operator]
    # normalize for cosine similarity
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    feats = feats / norms
    _index_embeddings = feats.astype(np.float32)
    _index_labels = labels
    _index_classes = classes
    _indexed_root = dataset_root
    return True


def ensure_index(dataset_root: str) -> bool:
    if _index_embeddings is not None and _indexed_root == dataset_root:
        return True
    return build_index(dataset_root)


def predict_food_local(uploaded_file, dataset_root: str) -> Optional[str]:
    """Predict class label by nearest neighbor in the local dataset index."""
    if not ensure_index(dataset_root):
        return None
    # Read bytes from Streamlit upload
    data = None
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
    else:
        data = uploaded_file
    try:
        arr = _load_image_to_array(data)
    except Exception:
        return None
    _ensure_model()
    if not _tf_ok or _feature_model is None:
        return None
    batch = np.expand_dims(arr, axis=0)
    feat = _feature_model(batch)  # type: ignore[operator]
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
    # cosine similarity against index
    sims = np.dot(_index_embeddings, feat.T).squeeze(1)  # type: ignore[arg-type]
    idx = int(np.argmax(sims))
    label = _index_labels[idx]
    return label.replace('_', ' ').title()


def get_index_status() -> dict:
    return {
        "ok": _index_embeddings is not None,
        "classes": len(_index_classes),
        "images_indexed": 0 if _index_embeddings is None else int(_index_embeddings.shape[0]),
        "root": _indexed_root,
        "error": _last_error,
    }

