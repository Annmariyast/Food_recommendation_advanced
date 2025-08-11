# utils/classify_image.py
import os
import io
import numpy as np
from PIL import Image

_tf_available = True
_model = None

def _load_tf_model_if_needed():
    global _tf_available, _model
    if not _tf_available:
        return
    if _model is not None:
        return
    try:
        # Silence TF logs
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions  # type: ignore
        _model = {
            'net': MobileNetV2(weights='imagenet'),
            'preprocess_input': preprocess_input,
            'decode_predictions': decode_predictions,
        }
    except Exception as _e:
        # TensorFlow not available or failed to load; mark unavailable
        _tf_available = False


def classify_food_image(uploaded_file):
    try:
        _load_tf_model_if_needed()
        if not _tf_available or _model is None:
            # Feature disabled: gracefully return None so UI can continue
            return None

        # Read bytes robustly (Streamlit may have advanced the file pointer)
        if hasattr(uploaded_file, 'getvalue'):
            data = uploaded_file.getvalue()
        elif hasattr(uploaded_file, 'read'):
            data = uploaded_file.read()
        else:
            data = uploaded_file  # assume bytes or path

        if isinstance(data, (bytes, bytearray)):
            img = Image.open(io.BytesIO(data)).convert('RGB')
        else:
            img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((224, 224))
        x = np.array(img, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        x = _model['preprocess_input'](x)

        preds = _model['net'].predict(x)
        decoded = _model['decode_predictions'](preds, top=3)[0]
        top_label = decoded[0][1].replace('_', ' ')
        return top_label
    except Exception as e:
        print("Image classification error:", e)
        return None


def get_image_classifier_status():
    try:
        status = {
            "tf_available": _tf_available,
            "model_loaded": _model is not None,
        }
        if _tf_available and _model is None:
            # try load once
            _load_tf_model_if_needed()
            status["model_loaded"] = _model is not None
        return status
    except Exception:
        return {"tf_available": False, "model_loaded": False}
