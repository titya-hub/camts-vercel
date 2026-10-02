const elements = {
  tabs: [...document.querySelectorAll(".mode-tab")],
  fileInput: document.querySelector("#file-input"),
  uploadBox: document.querySelector("#upload-box"),
  uploadTitle: document.querySelector("#upload-title"),
  uploadSubtitle: document.querySelector("#upload-subtitle"),
  fileNote: document.querySelector("#file-note"),
  browseButton: document.querySelector("#browse-button"),
  sampleGallery: document.querySelector("#sample-gallery"),
  sampleButtons: [...document.querySelectorAll(".sample-card")],
  cameraButton: document.querySelector("#camera-button"),
  mediaStage: document.querySelector("#media-stage"),
  emptyContent: document.querySelector("#empty-content"),
  mediaContent: document.querySelector("#media-content"),
  imageResult: document.querySelector("#image-result"),
  video: document.querySelector("#source-video"),
  canvas: document.querySelector("#overlay-canvas"),
  processingOverlay: document.querySelector("#processing-overlay"),
  processingLabel: document.querySelector("#processing-label"),
  resultSubtitle: document.querySelector("#result-subtitle"),
  resultBadge: document.querySelector("#result-badge"),
  detectionCount: document.querySelector("#detection-count"),
  detectionList: document.querySelector("#detection-list"),
  errorMessage: document.querySelector("#error-message"),
  recognitionStep: document.querySelector("#recognition-step"),
  pipelineNote: document.querySelector("#pipeline-note"),
};

const overlayContext = elements.canvas.getContext("2d");
let activeMode = "image";
let cameraStream = null;
let videoUrl = null;
let frameRequest = null;
let frameInProgress = false;
let lastFrameTime = 0;
let activeDetections = [];
let activeDetectionSize = null;
let mediaSession = 0;
let imageRequestId = 0;

function showError(message) {
  elements.errorMessage.textContent = message;
  elements.errorMessage.hidden = !message;
}

function clearResults() {
  elements.detectionCount.textContent = "—";
  elements.resultBadge.hidden = true;
  elements.detectionList.replaceChildren();
  const placeholder = document.createElement("p");
  placeholder.className = "list-placeholder";
  placeholder.textContent = "Detected signs and confidence scores will show here.";
  elements.detectionList.append(placeholder);
}

function setLoading(loading, label = "Analyzing image…") {
  elements.processingLabel.textContent = label;
  elements.processingOverlay.hidden = !loading;
}

function showMedia() {
  elements.emptyContent.hidden = true;
  elements.mediaContent.hidden = false;
  elements.mediaStage.classList.remove("empty-state");
}

function showEmptyState() {
  elements.emptyContent.hidden = false;
  elements.mediaContent.hidden = true;
  elements.mediaStage.classList.add("empty-state");
  elements.imageResult.hidden = true;
  elements.video.hidden = true;
  elements.canvas.hidden = true;
}

function setMode(mode) {
  if (activeMode === mode) return;
  mediaSession += 1;
  imageRequestId += 1;
  stopCamera();
  stopFrameLoop();
  activeMode = mode;
  activeDetections = [];
  activeDetectionSize = null;
  showError("");
  setLoading(false);
  clearResults();
  elements.imageResult.hidden = true;
  elements.video.hidden = true;
  elements.canvas.hidden = true;
  elements.video.onloadedmetadata = null;
  elements.video.onplay = null;
  elements.video.onpause = null;
  elements.video.onerror = null;
  elements.video.pause();
  elements.video.removeAttribute("src");
  elements.video.load();
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = null;
  showEmptyState();

  for (const tab of elements.tabs) {
    const selected = tab.dataset.mode === mode;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", String(selected));
  }

  const isImage = mode === "image";
  const isVideo = mode === "video";
  elements.fileInput.accept = isImage ? "image/*" : "video/*";
  elements.uploadBox.hidden = mode === "webcam";
  elements.sampleGallery.hidden = mode !== "image";
  elements.cameraButton.hidden = mode !== "webcam";
  elements.uploadTitle.textContent = isImage ? "Drop an image here" : "Drop a video here";
  elements.uploadSubtitle.textContent = isImage
    ? "or browse files from your device"
    : "or browse a video from your device";
  elements.browseButton.textContent = isImage ? "Choose image" : "Choose video";
  elements.fileNote.textContent = isImage
    ? "JPG, PNG or WEBP · up to 3 MB on Vercel"
    : "MP4, MOV or WEBM · analyzed in your browser";
  elements.recognitionStep.classList.toggle("disabled", !isImage);
  elements.recognitionStep.querySelector(".pipeline-check").textContent = isImage ? "✓" : "—";
  elements.pipelineNote.textContent = isImage
    ? "Both models run on uploaded images."
    : "Detection only for video and webcam frames.";
  elements.resultSubtitle.textContent = isImage
    ? "Your analyzed image will appear here."
    : isVideo
      ? "Play a video to see traffic sign detections."
      : "Start your webcam to see live detections.";
}

function percent(value) {
  return `${Math.round(value * 100)}%`;
}

function renderDetections(detections) {
  elements.detectionCount.textContent = String(detections.length);
  elements.resultBadge.textContent = `${detections.length} ${detections.length === 1 ? "sign" : "signs"}`;
  elements.resultBadge.hidden = false;
  elements.detectionList.replaceChildren();

  if (detections.length === 0) {
    const placeholder = document.createElement("p");
    placeholder.className = "list-placeholder";
    placeholder.textContent = "No traffic signs detected in this frame.";
    elements.detectionList.append(placeholder);
    return;
  }

  for (const detection of detections) {
    const card = document.createElement("div");
    card.className = "detection-card";
    const name = document.createElement("div");
    name.className = "detection-name";
    name.textContent = detection.class_name;

    if (detection.class_name !== detection.detection_class) {
      const detectorClass = document.createElement("span");
      detectorClass.className = "detection-type";
      detectorClass.textContent = `Detected as ${detection.detection_class}`;
      name.append(detectorClass);
    }

    const confidences = document.createElement("div");
    confidences.className = "confidence-group";
    const detectionConfidence = document.createElement("span");
    detectionConfidence.textContent = `Detect ${percent(detection.detection_confidence)}`;
    confidences.append(detectionConfidence);
    if (typeof detection.recognition_confidence === "number") {
      const recognitionConfidence = document.createElement("span");
      recognitionConfidence.className = "recognition-confidence";
      recognitionConfidence.textContent = `Recognize ${percent(detection.recognition_confidence)}`;
      confidences.append(recognitionConfidence);
    }
    card.append(name, confidences);
    elements.detectionList.append(card);
  }
}

async function readJsonResponse(response) {
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "The analysis request failed.");
  return payload;
}

async function analyzeImage(file) {
  const requestId = ++imageRequestId;
  const session = mediaSession;
  showError("");
  clearResults();
  showMedia();
  activeDetections = [];
  activeDetectionSize = null;
  elements.video.hidden = true;
  elements.canvas.hidden = true;
  elements.imageResult.hidden = true;
  setLoading(true);
  elements.resultSubtitle.textContent = `Analyzing ${file.name}…`;
  const form = new FormData();
  form.append("image", file);

  try {
    const response = await fetch("/api/detect-image", { method: "POST", body: form });
    const result = await readJsonResponse(response);
    if (requestId !== imageRequestId || session !== mediaSession || activeMode !== "image") return;
    elements.imageResult.src = result.image;
    elements.imageResult.hidden = false;
    renderDetections(result.detections);
    elements.resultSubtitle.textContent = `Analysis complete · ${file.name}`;
  } catch (error) {
    if (requestId !== imageRequestId || session !== mediaSession) return;
    showError(error.message);
    elements.resultSubtitle.textContent = "Could not analyze this image.";
  } finally {
    if (requestId === imageRequestId && session === mediaSession) setLoading(false);
  }
}

async function analyzeSample(button) {
  if (activeMode !== "image") return;

  const session = mediaSession;
  for (const sampleButton of elements.sampleButtons) sampleButton.disabled = true;
  showError("");

  try {
    const filename = button.dataset.sample;
    const response = await fetch(`/static/${encodeURIComponent(filename)}`);
    if (!response.ok) throw new Error("The sample image could not be loaded. Please try again.");

    const image = await response.blob();
    if (session !== mediaSession || activeMode !== "image") return;
    await analyzeImage(new File([image], filename, { type: image.type || "image/jpeg" }));
  } catch (error) {
    if (session === mediaSession) showError(error.message);
  } finally {
    for (const sampleButton of elements.sampleButtons) sampleButton.disabled = false;
  }
}

function selectVideo(file) {
  mediaSession += 1;
  stopFrameLoop();
  activeDetections = [];
  activeDetectionSize = null;
  showError("");
  clearResults();
  showMedia();
  elements.imageResult.hidden = true;
  elements.video.hidden = false;
  elements.canvas.hidden = false;
  elements.video.pause();
  elements.video.removeAttribute("src");
  elements.video.load();
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = URL.createObjectURL(file);
  elements.video.src = videoUrl;
  elements.resultSubtitle.textContent = `Ready to analyze · ${file.name}`;
  elements.video.onloadedmetadata = () => {
    resizeCanvas();
    drawDetections([]);
  };
  elements.video.onplay = () => {
    elements.resultSubtitle.textContent = `Analyzing video · ${file.name}`;
    startFrameLoop();
  };
  elements.video.onpause = () => {
    if (activeMode === "video") stopFrameLoop();
  };
  elements.video.onerror = () => showError("This video format could not be played by your browser.");
}

function resizeCanvas() {
  if (!elements.video.videoWidth || !elements.video.videoHeight) return;
  elements.canvas.width = elements.video.videoWidth;
  elements.canvas.height = elements.video.videoHeight;
  drawDetections(activeDetections, activeDetectionSize);
}

function drawDetections(detections, frameSize = null) {
  activeDetections = detections;
  activeDetectionSize = frameSize;
  if (!elements.canvas.width || !elements.canvas.height) return;
  const width = elements.canvas.width;
  const height = elements.canvas.height;
  overlayContext.clearRect(0, 0, width, height);
  const scaleX = frameSize ? width / frameSize.width : 1;
  const scaleY = frameSize ? height / frameSize.height : 1;
  const fontSize = Math.max(14, Math.round(width / 65));
  overlayContext.font = `600 ${fontSize}px "DM Sans", sans-serif`;
  overlayContext.lineWidth = Math.max(3, Math.round(width / 400));

  for (const detection of detections) {
    const [rawX1, rawY1, rawX2, rawY2] = detection.box;
    const x1 = rawX1 * scaleX;
    const y1 = rawY1 * scaleY;
    const x2 = rawX2 * scaleX;
    const y2 = rawY2 * scaleY;
    const label = `${detection.class_name}  ${percent(detection.detection_confidence)}`;
    const labelHeight = fontSize + 14;
    const labelWidth = overlayContext.measureText(label).width + 18;
    const labelY = Math.max(0, y1 - labelHeight);
    overlayContext.strokeStyle = "#129d3f";
    overlayContext.fillStyle = "#129d3f";
    overlayContext.strokeRect(x1, y1, x2 - x1, y2 - y1);
    overlayContext.fillRect(x1, labelY, labelWidth, labelHeight);
    overlayContext.fillStyle = "#fff";
    overlayContext.fillText(label, x1 + 9, labelY + fontSize + 1);
  }
}

function stopFrameLoop() {
  if (frameRequest !== null) cancelAnimationFrame(frameRequest);
  frameRequest = null;
  frameInProgress = false;
}

function startFrameLoop() {
  stopFrameLoop();
  lastFrameTime = 0;
  frameRequest = requestAnimationFrame(processVideoFrame);
}

async function processVideoFrame(timestamp) {
  if (activeMode === "video" && !elements.video.paused && !elements.video.ended) {
    frameRequest = requestAnimationFrame(processVideoFrame);
    if (timestamp - lastFrameTime >= 300 && !frameInProgress) {
      lastFrameTime = timestamp;
      frameInProgress = true;
      try {
        await analyzeCurrentFrame(elements.video);
      } catch (error) {
        showError(error.message);
        stopFrameLoop();
        return;
      } finally {
        frameInProgress = false;
      }
    }
  }
}

async function analyzeCurrentFrame(video) {
  if (!video.videoWidth || !video.videoHeight) return;
  const session = mediaSession;
  const maxWidth = 960;
  const scale = Math.min(1, maxWidth / video.videoWidth);
  const frameCanvas = document.createElement("canvas");
  frameCanvas.width = Math.round(video.videoWidth * scale);
  frameCanvas.height = Math.round(video.videoHeight * scale);
  frameCanvas.getContext("2d").drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
  const image = frameCanvas.toDataURL("image/jpeg", 0.78);
  const response = await fetch("/api/detect-frame", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image }),
  });
  const result = await readJsonResponse(response);
  if (
    session !== mediaSession ||
    (activeMode !== "webcam" && activeMode !== "video")
  ) return;
  drawDetections(result.detections, { width: result.width, height: result.height });
  renderDetections(result.detections);
}

async function startCamera() {
  if (cameraStream) {
    mediaSession += 1;
    stopFrameLoop();
    stopCamera();
    showEmptyState();
    clearResults();
    elements.resultSubtitle.textContent = "Start your webcam to see live detections.";
    return;
  }
  mediaSession += 1;
  activeDetections = [];
  activeDetectionSize = null;
  showError("");
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "environment" } },
    });
    if (activeMode !== "webcam") {
      stopCamera();
      return;
    }
    elements.video.srcObject = cameraStream;
    elements.video.hidden = false;
    elements.canvas.hidden = false;
    elements.imageResult.hidden = true;
    showMedia();
    await elements.video.play();
    resizeCanvas();
    elements.video.onloadedmetadata = () => {
      resizeCanvas();
      startFrameLoop();
    };
    startFrameLoop();
    elements.cameraButton.innerHTML = '<span class="camera-live-dot"></span><span>Stop webcam</span>';
    elements.resultSubtitle.textContent = "Live traffic sign detection · recognition off.";
  } catch (error) {
    showError(
      error.name === "NotAllowedError"
        ? "Camera access was denied. Allow camera access in your browser and try again."
        : `Could not start the webcam: ${error.message}`,
    );
  }
}

function stopCamera() {
  if (cameraStream) {
    for (const track of cameraStream.getTracks()) track.stop();
    cameraStream = null;
  }
  if (elements.video.srcObject) elements.video.srcObject = null;
  elements.cameraButton.innerHTML = '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 7.5A1.5 1.5 0 0 1 5.5 6h2l1.3-2h6.4l1.3 2h2A1.5 1.5 0 0 1 20 7.5v10a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17.5v-10Z" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="12.5" r="3.2" stroke="currentColor" stroke-width="1.6"/></svg><span>Start webcam</span>';
}

elements.tabs.forEach((tab) => tab.addEventListener("click", () => setMode(tab.dataset.mode)));
elements.browseButton.addEventListener("click", () => elements.fileInput.click());
elements.sampleButtons.forEach((button) => {
  button.addEventListener("click", () => analyzeSample(button));
});
elements.fileInput.addEventListener("change", () => {
  const file = elements.fileInput.files?.[0];
  if (!file) return;
  if (activeMode === "image") analyzeImage(file);
  else if (activeMode === "video") selectVideo(file);
  elements.fileInput.value = "";
});
elements.uploadBox.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.uploadBox.classList.add("dragging");
});
elements.uploadBox.addEventListener("dragleave", () => elements.uploadBox.classList.remove("dragging"));
elements.uploadBox.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.uploadBox.classList.remove("dragging");
  const file = event.dataTransfer.files[0];
  if (!file) return;
  if (activeMode === "image" && file.type.startsWith("image/")) analyzeImage(file);
  else if (activeMode === "video" && file.type.startsWith("video/")) selectVideo(file);
  else showError(`Choose a ${activeMode} file for the selected input mode.`);
});
elements.cameraButton.addEventListener("click", startCamera);
elements.video.addEventListener("loadedmetadata", resizeCanvas);
window.addEventListener("resize", resizeCanvas);
