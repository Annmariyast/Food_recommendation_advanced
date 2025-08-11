import os
from typing import List, Optional

import numpy as np
from PIL import Image

_tf_available = True
_tfds_available = True
_model = None
_labels: List[str] = []
_error: Optional[str] = None


def _models_dir() -> str:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_path = os.path.join(base_dir, "models")
    os.makedirs(models_path, exist_ok=True)
    return models_path


def _label_path() -> str:
    return os.path.join(_models_dir(), "food101_labels.txt")


def _model_path() -> str:
    return os.path.join(_models_dir(), "food101_efficientnet.keras")


def _load_labels() -> List[str]:
    global _tfds_available
    # Try cached labels file first
    labels_file = _label_path()
    if os.path.exists(labels_file):
        with open(labels_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    # Fallback: fetch from TFDS metadata (no full dataset download)
    try:
        import tensorflow_datasets as tfds  # type: ignore

        builder = tfds.builder("food101")
        builder.download_and_prepare(download_config=tfds.download.DownloadConfig(try_gcs=True))
        names = list(builder.info.features["label"].names)
        # Cache to disk
        with open(labels_file, "w", encoding="utf-8") as f:
            for name in names:
                f.write(f"{name}\n")
        return names
    except Exception:
        _tfds_available = False
        return []


def _build_model(num_classes: int):
    global _tf_available
    try:
        # Silence TF logs
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf  # type: ignore
        from tensorflow.keras import layers, models  # type: ignore
        from tensorflow.keras.applications import EfficientNetB0  # type: ignore
        from tensorflow.keras.applications.efficientnet import preprocess_input  # type: ignore

        base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        base.trainable = False
        inputs = layers.Input(shape=(224, 224, 3))
        x = preprocess_input(inputs)
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = models.Model(inputs, outputs)
        return model
    except Exception:
        _tf_available = False
        return None


def load_food101_classifier():
    """Load a Food-101 EfficientNet classifier if available on disk.

    Returns True if ready to predict; False otherwise.
    """
    global _model, _labels, _error, _tf_available
    if _model is not None and _labels:
        return True

    model_file = _model_path()
    try:
        # Only proceed if a saved model exists locally. Avoid TFDS downloads by default.
        if os.path.exists(model_file):
            import tensorflow as tf  # type: ignore
            _labels = _load_labels()
            if not _labels:
                _error = "Food-101 labels file missing. Upload labels in Advanced section."
                return False
            _model = tf.keras.models.load_model(model_file)
            return True

        _error = (
            f"No saved Food-101 model found at {model_file}. Upload a .keras/.h5 model in Advanced."
        )
        return False
    except Exception as e:
        _error = f"Failed to load Food-101 model: {e}"
        return False


def predict_food101(uploaded_file) -> Optional[str]:
    """Predict food label using Food-101 classifier. Returns None if unavailable."""
    global _model, _labels
    if _model is None or not _labels:
        if not load_food101_classifier():
            return None

    try:
        # Preprocess image
        img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
        x = np.asarray(img, dtype=np.float32)
        x = np.expand_dims(x, axis=0)

        preds = _model.predict(x)
        idx = int(np.argmax(preds, axis=1)[0])
        if 0 <= idx < len(_labels):
            return _labels[idx].replace("_", " ")
        return None
    except Exception:
        return None


def get_food101_status() -> str:
    if _model is not None and _labels:
        return "ready"
    if _error:
        return _error
    return "not_initialized"


def set_food101_model_from_upload(file_like) -> bool:
    """Accepts an uploaded model file (.keras or .h5), saves and loads it."""
    global _model, _error
    try:
        import tensorflow as tf  # type: ignore
        target = _model_path()
        # Save bytes to disk
        data = file_like.read() if hasattr(file_like, "read") else file_like
        if isinstance(data, (bytes, bytearray)):
            with open(target, "wb") as f:
                f.write(data)
        else:
            # Fallback: stream chunks
            with open(target, "wb") as f:
                while True:
                    chunk = file_like.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
        # Try loading the model to ensure it is valid
        _model = tf.keras.models.load_model(target)
        _error = None
        return True
    except Exception as e:
        _error = f"Invalid Food-101 model: {e}"
        return False


def set_food101_labels_from_upload(file_like) -> bool:
    """Accepts a labels text file with 101 class names (one per line)."""
    global _labels
    try:
        data = file_like.read() if hasattr(file_like, "read") else file_like
        text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)
        names = [line.strip() for line in text.splitlines() if line.strip()]
        if len(names) < 2:
            return False
        with open(_label_path(), "w", encoding="utf-8") as f:
            for n in names:
                f.write(f"{n}\n")
        _labels = names
        return True
    except Exception:
        return False

