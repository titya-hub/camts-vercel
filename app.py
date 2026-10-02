import base64
import ast
import binascii
import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
DETECTOR_PATH = Path(
    os.environ.get(
        "CAMTS_DETECTOR_MODEL",
        str(BASE_DIR / "models" / "camtsd1_8n_int8.onnx"),
    )
)
CLASSIFIER_PATH = Path(
    os.environ.get(
        "CAMTS_CLASSIFIER_MODEL",
        str(BASE_DIR / "models" / "camtsr.onnx"),
    )
)
DETECTION_SIZE = 640
CLASSIFICATION_SIZE = 64
CLASSIFICATION_RESIZE = round(CLASSIFICATION_SIZE / 0.875)
DETECTION_CONFIDENCE = 0.25
DETECTION_IOU = 0.45
MAX_IMAGE_BYTES = 3 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_RESPONSE_IMAGE_BYTES = 3 * 1024 * 1024
BRAND_GREEN = (63, 157, 18)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES + 256 * 1024
app.config["MAX_IMAGE_PIXELS"] = MAX_IMAGE_PIXELS
app.logger.setLevel(logging.INFO)

_detector = None
_classifier = None
_detector_names = {}
_classifier_names = {}
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


class ModelUnavailableError(RuntimeError):
    pass


def _load_model(path, model_type, expected_input, expected_output):
    if not path.is_file():
        raise ModelUnavailableError(f"The {model_type} model was not found: {path}")

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ModelUnavailableError(
            "ONNX Runtime is not installed. Install the project dependencies first."
        ) from error

    try:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape
        if list(input_shape) != expected_input or list(output_shape) != expected_output:
            raise ModelUnavailableError(
                f"The {model_type} ONNX model has incompatible input/output shapes: "
                f"{input_shape} -> {output_shape}."
            )

        metadata_names = session.get_modelmeta().custom_metadata_map.get("names", "{}")
        names = ast.literal_eval(metadata_names)
        return session, {int(key): str(value) for key, value in names.items()}
    except ModelUnavailableError:
        raise
    except (OSError, RuntimeError, ValueError, SyntaxError) as error:
        raise ModelUnavailableError(
            f"Could not load the {model_type} ONNX model: {error}"
        ) from error


def get_detector():
    global _detector, _detector_names
    if _detector is None:
        with _model_lock:
            if _detector is None:
                _detector, _detector_names = _load_model(
                    DETECTOR_PATH,
                    "detection",
                    [1, 3, DETECTION_SIZE, DETECTION_SIZE],
                    [
                        1,
                        5,
                        (DETECTION_SIZE // 8) ** 2
                        + (DETECTION_SIZE // 16) ** 2
                        + (DETECTION_SIZE // 32) ** 2,
                    ],
                )
    return _detector


def get_classifier():
    global _classifier, _classifier_names
    if _classifier is None:
        with _model_lock:
            if _classifier is None:
                _classifier, _classifier_names = _load_model(
                    CLASSIFIER_PATH,
                    "recognition",
                    [1, 3, CLASSIFICATION_SIZE, CLASSIFICATION_SIZE],
                    [1, 40],
                )
    return _classifier


def decode_image(image_bytes):
    if not image_bytes:
        raise ValueError("Choose a non-empty image.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Images must be 3 MB or smaller on this Vercel trial.")

    image = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError("The selected file is not a supported image.")
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError("Images must be 25 megapixels or smaller.")
    return image


def decode_data_url(data_url):
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("A valid camera frame is required.")

    header, encoded_image = data_url.split(",", 1)
    if header not in (
        "data:image/jpeg;base64",
        "data:image/png;base64",
        "data:image/webp;base64",
    ):
        raise ValueError("Camera frames must be JPEG, PNG, or WebP images.")

    try:
        image_bytes = base64.b64decode(encoded_image, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("The camera frame could not be decoded.") from error

    return decode_image(image_bytes)


def _letterbox(image):
    height, width = image.shape[:2]
    scale = min(DETECTION_SIZE / height, DETECTION_SIZE / width)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    pad_width = DETECTION_SIZE - resized_width
    pad_height = DETECTION_SIZE - resized_height
    left = round(pad_width / 2 - 0.1)
    right = round(pad_width / 2 + 0.1)
    top = round(pad_height / 2 - 0.1)
    bottom = round(pad_height / 2 + 0.1)
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    ), scale, left, top


def _non_max_suppression(boxes, scores):
    if not len(scores):
        return []

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = np.argsort(scores)[::-1]
    selected = []

    while order.size:
        current = int(order[0])
        selected.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        intersection_width = np.maximum(
            0, np.minimum(x2[current], x2[rest]) - np.maximum(x1[current], x1[rest])
        )
        intersection_height = np.maximum(
            0, np.minimum(y2[current], y2[rest]) - np.maximum(y1[current], y1[rest])
        )
        intersection = intersection_width * intersection_height
        union = areas[current] + areas[rest] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = rest[iou <= DETECTION_IOU]

    return selected


def _prepare_classification_crop(crop):
    height, width = crop.shape[:2]
    scale = CLASSIFICATION_RESIZE / min(height, width)
    resized_width = max(CLASSIFICATION_RESIZE, int(width * scale))
    resized_height = max(CLASSIFICATION_RESIZE, int(height * scale))
    resized = cv2.resize(
        crop,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    left = (resized_width - CLASSIFICATION_SIZE) // 2
    top = (resized_height - CLASSIFICATION_SIZE) // 2
    crop = resized[
        top : top + CLASSIFICATION_SIZE, left : left + CLASSIFICATION_SIZE
    ]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]


def detect(image, recognize=False):
    height, width = image.shape[:2]
    detector = get_detector()
    letterboxed, scale, pad_x, pad_y = _letterbox(image)
    detector_input = np.transpose(
        cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
        (2, 0, 1),
    )[None]
    output = detector.run(
        [detector.get_outputs()[0].name],
        {detector.get_inputs()[0].name: detector_input},
    )[0][0]
    predictions = output.T
    scores = predictions[:, 4]
    candidate_indices = np.flatnonzero(scores >= DETECTION_CONFIDENCE)
    detections = []
    if not candidate_indices.size:
        return detections

    selected = predictions[candidate_indices]
    centers_x, centers_y, widths, heights = selected[:, :4].T
    boxes = np.column_stack(
        (
            centers_x - widths / 2,
            centers_y - heights / 2,
            centers_x + widths / 2,
            centers_y + heights / 2,
        )
    )
    keep = _non_max_suppression(boxes, selected[:, 4])
    classifier = get_classifier() if recognize else None
    classifier_input_name = classifier.get_inputs()[0].name if classifier else None
    classifier_output_name = classifier.get_outputs()[0].name if classifier else None

    for index in keep:
        box = boxes[index]
        x1 = max(0, min(width, int(round((box[0] - pad_x) / scale))))
        y1 = max(0, min(height, int(round((box[1] - pad_y) / scale))))
        x2 = max(0, min(width, int(round((box[2] - pad_x) / scale))))
        y2 = max(0, min(height, int(round((box[3] - pad_y) / scale))))
        if x2 <= x1 or y2 <= y1:
            continue

        class_id = int(selected[index, 5]) if selected.shape[1] > 5 else 0
        detection_class = _detector_names.get(class_id, str(class_id))
        confidence = float(selected[index, 4])
        detection = {
            "box": [x1, y1, x2, y2],
            "detection_class": detection_class,
            "detection_confidence": confidence,
            "class_name": str(detection_class),
        }

        if classifier is not None:
            crop = image[y1:y2, x1:x2]
            classifier_input = _prepare_classification_crop(crop)
            probabilities = classifier.run(
                [classifier_output_name],
                {classifier_input_name: classifier_input},
            )[0][0]
            class_index = int(np.argmax(probabilities))
            detection["class_name"] = _classifier_names.get(
                class_index, str(class_index)
            )
            detection["recognition_confidence"] = float(probabilities[class_index])

        detections.append(detection)

    return detections


def annotate_image(image, detections):
    annotated = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BRAND_GREEN, 3)
        label = f'{detection["class_name"]}  {detection["detection_confidence"]:.0%}'
        if "recognition_confidence" in detection:
            label += f'  |  {detection["recognition_confidence"]:.0%}'

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(image.shape[1] / 1600, 0.8))
        (label_width, label_height), baseline = cv2.getTextSize(
            label, font, font_scale, 2
        )
        label_top = max(0, y1 - label_height - baseline - 10)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (min(image.shape[1], x1 + label_width + 12), y1),
            BRAND_GREEN,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 6, max(label_height, y1 - baseline - 5)),
            font,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def encode_annotated_image(image):
    height, width = image.shape[:2]
    scale = min(1.0, 1600 / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    for _ in range(5):
        success, encoded_image = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if not success:
            raise RuntimeError("OpenCV could not encode the annotated image.")
        if len(encoded_image) <= MAX_RESPONSE_IMAGE_BYTES:
            return encoded_image

        height, width = image.shape[:2]
        if max(height, width) <= 640:
            break
        image = cv2.resize(
            image,
            (max(1, int(width * 0.8)), max(1, int(height * 0.8))),
            interpolation=cv2.INTER_AREA,
        )

    success, encoded_image = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70]
    )
    if not success or len(encoded_image) > MAX_RESPONSE_IMAGE_BYTES:
        raise RuntimeError(
            "The annotated image is too large for a Vercel function response."
        )
    return encoded_image


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "detector_model": DETECTOR_PATH.is_file(),
            "classifier_model": CLASSIFIER_PATH.is_file(),
            "ready": DETECTOR_PATH.is_file() and CLASSIFIER_PATH.is_file(),
        }
    )


@app.post("/api/detect-image")
def detect_image():
    upload = request.files.get("image")
    if upload is None:
        return jsonify(error="Choose an image to analyze."), 400

    try:
        image = decode_image(upload.read(MAX_IMAGE_BYTES + 1))
    except ValueError as error:
        return jsonify(error=str(error)), 400

    try:
        with _inference_lock:
            detections = detect(image, recognize=True)
    except ModelUnavailableError as error:
        app.logger.error("%s", error)
        return jsonify(error=str(error)), 503
    except (OSError, RuntimeError, ValueError) as error:
        app.logger.exception("Image inference failed")
        return jsonify(error=f"Image inference failed: {error}"), 500

    try:
        encoded_image = encode_annotated_image(annotate_image(image, detections))
    except RuntimeError as error:
        app.logger.exception("Could not prepare the annotated image response")
        return jsonify(error=str(error)), 500

    image_url = "data:image/jpeg;base64," + base64.b64encode(
        encoded_image.tobytes()
    ).decode("ascii")
    return jsonify(detections=detections, image=image_url)


@app.post("/api/detect-frame")
def detect_frame():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="A camera or video frame is required."), 400

    try:
        image = decode_data_url(payload.get("image"))
    except ValueError as error:
        return jsonify(error=str(error)), 400

    try:
        with _inference_lock:
            detections = detect(image)
    except ModelUnavailableError as error:
        app.logger.error("%s", error)
        return jsonify(error=str(error)), 503
    except (OSError, RuntimeError, ValueError) as error:
        app.logger.exception("Frame inference failed")
        return jsonify(error=f"Frame inference failed: {error}"), 500

    return jsonify(
        detections=detections,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
    )


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error="The upload is too large. Choose a smaller image or frame."), 413


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        threaded=True,
    )
