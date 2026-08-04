import * as THREE from "three";
import { OrbitControls } from "/static/vendor/OrbitControls.js";

const state = {
  systemKey: "her2_experimental",
  methodKey: "direct",
  systems: new Map(),
  selectedFile: null,
  result: null,
  structure: null,
  targetVisible: false,
  colorByDisplacement: false,
  motionValue: 0,
  playing: false,
  playDirection: 1,
  activeExample: null,
  webglAvailable: true,
  fallbackRotationX: -0.18,
  fallbackRotationY: 0.46,
  fallbackZoom: 1,
};

const elements = Object.fromEntries(
  [
    "service-status", "file-input", "drop-zone", "input-preview-wrap", "input-preview",
    "input-name", "input-shape", "predict-button", "reset-button", "examples-grid",
    "viewer-panel", "viewer", "empty-state", "loading-state", "viewer-toolbar", "motion-controls",
    "motion-slider", "motion-value", "play-button", "target-toggle", "fit-button",
    "fullscreen-button", "color-mode", "chain-legend", "displacement-legend", "legend-title",
    "result-title", "result-subtitle", "metric-list", "comparison-callout",
    "preprocess-details", "download-button", "toast", "fact-output", "empty-description",
    "evidence-stats", "evidence-conclusion", "system-caveat", "method-normalization",
    "method-output-title", "method-output", "method-title", "system-selector", "method-selector",
    "method-stage-two-title", "method-stage-two-detail", "method-stage-three-title",
    "method-stage-three-detail", "fact-branch", "fact-encoder",
  ].map((id) => [id, document.getElementById(id)])
);

let scene;
let camera;
let renderer;
let controls;
let structureGroup;
let targetGroup;
let atomPoints;
let bondLines;
let fallbackCanvas;
let fallbackContext;
let animationPrevious = performance.now();

function initializeViewer() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xeef3f4);
  camera = new THREE.PerspectiveCamera(38, 1, 0.1, 3000);
  camera.position.set(0, 0, 240);
  renderer = new THREE.WebGLRenderer({
    antialias: false,
    alpha: false,
    powerPreference: "high-performance",
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.addEventListener("webglcontextlost", (event) => {
    event.preventDefault();
    activateFallbackRenderer();
  });
  elements.viewer.appendChild(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.55;
  controls.panSpeed = 0.5;
  controls.minDistance = 25;
  controls.maxDistance = 800;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa8ad, 2.4));
  const directional = new THREE.DirectionalLight(0xffffff, 2.1);
  directional.position.set(1, 1.4, 2);
  scene.add(directional);
  resizeViewer();
  window.addEventListener("resize", resizeViewer);
  requestAnimationFrame(renderLoop);
}

function resizeViewer() {
  const bounds = elements.viewer.getBoundingClientRect();
  if (bounds.width <= 0 || bounds.height <= 0) return;
  camera.aspect = bounds.width / bounds.height;
  camera.updateProjectionMatrix();
  renderer.setSize(bounds.width, bounds.height, false);
  if (fallbackCanvas) {
    const scale = Math.min(window.devicePixelRatio, 2);
    fallbackCanvas.width = Math.round(bounds.width * scale);
    fallbackCanvas.height = Math.round(bounds.height * scale);
    fallbackCanvas.style.width = `${bounds.width}px`;
    fallbackCanvas.style.height = `${bounds.height}px`;
    drawFallbackStructure();
  }
}

function renderLoop(now) {
  const delta = Math.min((now - animationPrevious) / 1000, 0.1);
  animationPrevious = now;
  if (state.playing && state.result) {
    state.motionValue += delta * 26 * state.playDirection;
    if (state.motionValue >= 100) {
      state.motionValue = 100;
      state.playDirection = -1;
    } else if (state.motionValue <= 0) {
      state.motionValue = 0;
      state.playDirection = 1;
    }
    elements["motion-slider"].value = String(state.motionValue);
    updateMotionDisplay();
  }
  controls.update();
  if (state.webglAvailable) renderer.render(scene, camera);
  requestAnimationFrame(renderLoop);
}

function activateFallbackRenderer() {
  if (!state.webglAvailable && fallbackCanvas) return;
  state.webglAvailable = false;
  renderer.domElement.style.display = "none";
  fallbackCanvas = document.createElement("canvas");
  fallbackCanvas.id = "fallback-canvas";
  fallbackCanvas.className = "fallback-canvas";
  fallbackCanvas.setAttribute("aria-label", "Fallback projected coordinate structure");
  elements.viewer.appendChild(fallbackCanvas);
  fallbackContext = fallbackCanvas.getContext("2d");
  let dragging = false;
  let previousX = 0;
  let previousY = 0;
  fallbackCanvas.addEventListener("pointerdown", (event) => {
    dragging = true;
    previousX = event.clientX;
    previousY = event.clientY;
    fallbackCanvas.setPointerCapture(event.pointerId);
  });
  fallbackCanvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    state.fallbackRotationY += (event.clientX - previousX) * 0.008;
    state.fallbackRotationX += (event.clientY - previousY) * 0.008;
    previousX = event.clientX;
    previousY = event.clientY;
    drawFallbackStructure();
  });
  fallbackCanvas.addEventListener("pointerup", () => { dragging = false; });
  fallbackCanvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.fallbackZoom = Math.max(0.55, Math.min(2.2, state.fallbackZoom * (event.deltaY > 0 ? 0.92 : 1.08)));
    drawFallbackStructure();
  }, { passive: false });
  resizeViewer();
  drawFallbackStructure();
}

function projectedFallbackCoordinates(coordinates) {
  if (!coordinates?.length || !fallbackCanvas) return [];
  const center = coordinates.reduce((sum, point) => [sum[0] + point[0], sum[1] + point[1], sum[2] + point[2]], [0, 0, 0]).map((value) => value / coordinates.length);
  const cosY = Math.cos(state.fallbackRotationY);
  const sinY = Math.sin(state.fallbackRotationY);
  const cosX = Math.cos(state.fallbackRotationX);
  const sinX = Math.sin(state.fallbackRotationX);
  const rotated = coordinates.map((point) => {
    const x = point[0] - center[0];
    const y = point[1] - center[1];
    const z = point[2] - center[2];
    const xY = x * cosY + z * sinY;
    const zY = -x * sinY + z * cosY;
    return [xY, y * cosX - zY * sinX, y * sinX + zY * cosX];
  });
  const extent = Math.max(...rotated.flatMap((point) => [Math.abs(point[0]), Math.abs(point[1])]), 1);
  const scale = Math.min(fallbackCanvas.width, fallbackCanvas.height) * 0.39 * state.fallbackZoom / extent;
  return rotated.map((point) => [fallbackCanvas.width / 2 + point[0] * scale, fallbackCanvas.height / 2 - point[1] * scale, point[2]]);
}

function drawFallbackStructure() {
  if (!fallbackContext || !state.result) return;
  const coordinates = currentCoordinates();
  const projected = projectedFallbackCoordinates(coordinates);
  const context = fallbackContext;
  context.clearRect(0, 0, fallbackCanvas.width, fallbackCanvas.height);
  context.fillStyle = "#eef3f4";
  context.fillRect(0, 0, fallbackCanvas.width, fallbackCanvas.height);
  if (state.targetVisible && state.result.paired_target) {
    const target = projectedFallbackCoordinates(state.result.paired_target);
    context.lineWidth = 1.4;
    context.strokeStyle = "rgba(232,106,51,.72)";
    for (const segment of state.result.segments) {
      context.beginPath();
      context.moveTo(target[segment[0]][0], target[segment[0]][1]);
      for (let index = 1; index < segment.length; index += 1) context.lineTo(target[segment[index]][0], target[segment[index]][1]);
      context.stroke();
    }
  }
  const displacement = state.result.displacement;
  const sorted = [...displacement].sort((a, b) => a - b);
  const upper = sorted[Math.floor(sorted.length * 0.95)] || 1;
  const atoms = state.result.topology.atoms;
  context.lineWidth = Math.max(1.2, fallbackCanvas.width / 900);
  context.lineJoin = "round";
  context.lineCap = "round";
  for (const segment of state.result.segments) {
    const chain = atoms[segment[0]].chain;
    context.strokeStyle = state.result.chain_colors[chain] || "#657584";
    context.beginPath();
    context.moveTo(projected[segment[0]][0], projected[segment[0]][1]);
    for (let index = 1; index < segment.length; index += 1) context.lineTo(projected[segment[index]][0], projected[segment[index]][1]);
    context.stroke();
  }
  const pointRadius = Math.max(1.15, fallbackCanvas.width / 720);
  projected.forEach((point, index) => {
    if (state.colorByDisplacement) {
      const color = displacementColor(displacement[index], upper);
      context.fillStyle = `rgb(${Math.round(color.r * 255)},${Math.round(color.g * 255)},${Math.round(color.b * 255)})`;
    } else {
      context.fillStyle = state.result.chain_colors[atoms[index].chain] || "#657584";
    }
    context.beginPath();
    context.arc(point[0], point[1], pointRadius, 0, Math.PI * 2);
    context.fill();
  });
}

function currentCoordinates() {
  if (!state.result) return null;
  const mean = state.result.training_mean;
  const prediction = state.result.prediction;
  const fraction = state.motionValue / 100;
  return prediction.map((point, index) => [
    mean[index][0] + (point[0] - mean[index][0]) * fraction,
    mean[index][1] + (point[1] - mean[index][1]) * fraction,
    mean[index][2] + (point[2] - mean[index][2]) * fraction,
  ]);
}

function displacementColor(value, upper) {
  const t = Math.max(0, Math.min(1, value / Math.max(upper, 1e-6)));
  const stops = [
    [0.0, new THREE.Color("#2f6db2")],
    [0.38, new THREE.Color("#45b8a6")],
    [0.72, new THREE.Color("#f0c44c")],
    [1.0, new THREE.Color("#d4472f")],
  ];
  for (let index = 1; index < stops.length; index += 1) {
    if (t <= stops[index][0]) {
      const [leftT, leftColor] = stops[index - 1];
      const [rightT, rightColor] = stops[index];
      return leftColor.clone().lerp(rightColor, (t - leftT) / (rightT - leftT));
    }
  }
  return stops.at(-1)[1].clone();
}

function createStructure() {
  if (structureGroup) scene.remove(structureGroup);
  if (targetGroup) scene.remove(targetGroup);
  structureGroup = new THREE.Group();
  targetGroup = new THREE.Group();
  scene.add(structureGroup);
  scene.add(targetGroup);
  const count = state.result.prediction.length;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
  const material = new THREE.PointsMaterial({ size: 2.1, vertexColors: true, sizeAttenuation: true });
  atomPoints = new THREE.Points(geometry, material);
  structureGroup.add(atomPoints);
  const bondGeometry = new THREE.BufferGeometry();
  const bondCount = state.result.segments.reduce((total, segment) => total + Math.max(segment.length - 1, 0), 0);
  bondGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(bondCount * 2 * 3), 3));
  const bondMaterial = new THREE.LineBasicMaterial({ color: 0x657584, transparent: true, opacity: 0.72 });
  bondLines = new THREE.LineSegments(bondGeometry, bondMaterial);
  structureGroup.add(bondLines);
  if (state.result.paired_target) buildTargetOverlay(state.result.paired_target);
  updateMotionDisplay();
  fitStructure();
}

function buildTargetOverlay(coordinates) {
  const positions = [];
  for (const segment of state.result.segments) {
    for (let index = 1; index < segment.length; index += 1) {
      positions.push(...coordinates[segment[index - 1]], ...coordinates[segment[index]]);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color: 0xe86a33, transparent: true, opacity: 0.78 });
  targetGroup.add(new THREE.LineSegments(geometry, material));
  targetGroup.visible = state.targetVisible;
}

function updateMotionDisplay() {
  if (!state.result || !atomPoints || !bondLines) return;
  const coordinates = currentCoordinates();
  const atoms = state.result.topology.atoms;
  const displacement = state.result.displacement;
  const sorted = [...displacement].sort((a, b) => a - b);
  const upper = sorted[Math.floor(sorted.length * 0.95)] || 1;
  const color = new THREE.Color();
  const atomPositions = atomPoints.geometry.attributes.position.array;
  const atomColors = atomPoints.geometry.attributes.color.array;
  coordinates.forEach((point, index) => {
    atomPositions.set(point, index * 3);
    if (state.colorByDisplacement) {
      color.copy(displacementColor(displacement[index], upper));
    } else {
      color.set(state.result.chain_colors[atoms[index].chain] || "#657584");
    }
    atomColors.set([color.r, color.g, color.b], index * 3);
  });
  atomPoints.geometry.attributes.position.needsUpdate = true;
  atomPoints.geometry.attributes.color.needsUpdate = true;
  atomPoints.geometry.computeBoundingSphere();
  const positions = bondLines.geometry.attributes.position.array;
  let cursor = 0;
  for (const segment of state.result.segments) {
    for (let index = 1; index < segment.length; index += 1) {
      const first = coordinates[segment[index - 1]];
      const second = coordinates[segment[index]];
      positions.set(first, cursor);
      positions.set(second, cursor + 3);
      cursor += 6;
    }
  }
  bondLines.geometry.attributes.position.needsUpdate = true;
  bondLines.geometry.computeBoundingSphere();
  elements["motion-value"].textContent = `${Math.round(state.motionValue)}%`;
  if (!state.webglAvailable) drawFallbackStructure();
}

function fitStructure() {
  if (!state.result) return;
  const coordinates = currentCoordinates();
  const bounds = new THREE.Box3();
  coordinates.forEach((point) => bounds.expandByPoint(new THREE.Vector3(...point)));
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) * 0.7;
  controls.target.copy(center);
  camera.position.set(center.x + radius * 0.75, center.y + radius * 0.35, center.z + radius * 1.55);
  camera.near = Math.max(0.1, radius / 100);
  camera.far = Math.max(1000, radius * 10);
  camera.updateProjectionMatrix();
  controls.update();
}

function setLoading(isLoading) {
  elements["loading-state"].classList.toggle("is-hidden", !isLoading);
  elements["predict-button"].disabled = isLoading || !state.selectedFile;
}

function showToast(message, isError = false) {
  elements.toast.textContent = message;
  elements.toast.classList.toggle("is-error", isError);
  elements.toast.classList.add("is-visible");
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(() => elements.toast.classList.remove("is-visible"), 3500);
}

async function apiRequest(payload) {
  setLoading(true);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...payload, system: state.systemKey, method: state.methodKey }),
    });
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      throw new Error(
        response.ok
          ? "The prediction service returned an unexpected response."
          : `The prediction service is unavailable (HTTP ${response.status}). Reload the application and try again.`,
      );
    }
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");
    applyResult(data);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setLoading(false);
  }
}

function applyResult(result) {
  state.result = result;
  state.motionValue = 100;
  state.playing = false;
  state.playDirection = -1;
  state.targetVisible = false;
  elements["motion-slider"].value = "100";
  elements["empty-state"].classList.add("is-hidden");
  elements["viewer-toolbar"].classList.remove("is-disabled");
  elements["motion-controls"].classList.remove("is-disabled");
  elements["download-button"].disabled = false;
  elements["target-toggle"].classList.toggle("is-hidden", !result.paired_target);
  elements["target-toggle"].classList.remove("is-active");
  setPlayIcon(false);
  createStructure();
  populateChainLegend();
  populateResultPanel();
  showToast("Coordinate prediction complete.");
}

function formatAngstrom(value) {
  return Number.isFinite(value) ? `${value.toFixed(3)} Å` : "—";
}

function populateChainLegend() {
  if (!state.result) return;
  const labels = state.result.topology.chain_labels || {};
  elements["chain-legend"].innerHTML = Object.entries(labels).map(([chain, label]) => (
    `<span><i style="--swatch:${state.result.chain_colors[chain] || "#657584"}"></i>${label}</span>`
  )).join("");
}

function populateResultPanel() {
  const metrics = state.result.metrics;
  const paired = Number.isFinite(metrics.paired_target_raw_rmsd);
  elements["result-title"].textContent = state.result.test_identifier === null
    ? state.result.source_name
    : `Held-out particle ${String(state.result.test_identifier).padStart(5, "0")}`;
  elements["result-subtitle"].textContent = paired
    ? `Prediction evaluated against its ${state.result.model.target_status}.`
    : "Uploaded image prediction; no paired target is available for accuracy evaluation.";
  elements["metric-list"].innerHTML = `
    <div class="metric-row"><span>Paired raw RMSD</span><strong>${paired ? formatAngstrom(metrics.paired_target_raw_rmsd) : "Not available"}</strong></div>
    <div class="metric-row"><span>Displacement from training mean</span><strong>${formatAngstrom(metrics.displacement_from_training_mean_rmsd)}</strong></div>
    <div class="metric-row"><span>Prediction radius of gyration</span><strong>${formatAngstrom(metrics.prediction_radius_of_gyration)}</strong></div>
    <div class="metric-row"><span>p95 positional displacement</span><strong>${formatAngstrom(metrics.p95_position_displacement)}</strong></div>
  `;
  if (paired) {
    const improved = metrics.improves_on_training_mean;
    elements["comparison-callout"].classList.remove("is-hidden");
    elements["comparison-callout"].classList.toggle("is-worse", !improved);
    elements["comparison-callout"].textContent = improved
      ? `This prediction improves on the fixed training-target mean (${formatAngstrom(metrics.training_mean_raw_rmsd)}).`
      : `This prediction does not improve on the fixed training-target mean (${formatAngstrom(metrics.training_mean_raw_rmsd)}).`;
  } else {
    elements["comparison-callout"].classList.add("is-hidden");
  }
  const processing = state.result.preprocessing;
  elements["preprocess-details"].innerHTML = `
    <div><dt>Source</dt><dd>${processing.source || "uploaded image"}</dd></div>
    <div><dt>Input shape</dt><dd>${processing.original_shape.join(" × ")}</dd></div>
    <div><dt>Model shape</dt><dd>128 × 128</dd></div>
    <div><dt>Scaling</dt><dd>${processing.normalization}</dd></div>
  `;
  if (processing.shape_warning) showToast(processing.shape_warning);
}

function fileToPayload(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const bytes = new Uint8Array(reader.result);
      let binary = "";
      const chunk = 0x8000;
      for (let start = 0; start < bytes.length; start += chunk) {
        binary += String.fromCharCode(...bytes.subarray(start, start + chunk));
      }
      resolve({ filename: file.name, content_base64: btoa(binary) });
    };
    reader.onerror = () => reject(new Error("Could not read the selected file."));
    reader.readAsArrayBuffer(file);
  });
}

function selectFile(file) {
  if (!file) return;
  state.selectedFile = file;
  state.activeExample = null;
  document.querySelectorAll(".example-button").forEach((button) => button.classList.remove("is-active"));
  elements["input-preview-wrap"].classList.remove("is-hidden");
  elements["input-name"].textContent = file.name;
  elements["input-shape"].textContent = `${(file.size / 1024).toFixed(1)} KB`;
  elements["predict-button"].disabled = false;
  if (file.type.startsWith("image/")) {
    elements["input-preview"].src = URL.createObjectURL(file);
  } else {
    elements["input-preview"].src = "/static/npy-placeholder.svg";
  }
}

function resetInput() {
  state.selectedFile = null;
  state.activeExample = null;
  elements["file-input"].value = "";
  elements["input-preview-wrap"].classList.add("is-hidden");
  elements["predict-button"].disabled = true;
  document.querySelectorAll(".example-button").forEach((button) => button.classList.remove("is-active"));
}

async function loadExamples() {
  try {
    const requestedSystem = state.systemKey;
    const response = await fetch(`/api/examples?system=${encodeURIComponent(requestedSystem)}`);
    const data = await response.json();
    if (requestedSystem !== state.systemKey) return;
    elements["examples-grid"].innerHTML = "";
    data.examples.forEach((example) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "example-button";
      button.title = example.label;
      button.innerHTML = `<img src="${example.thumbnail}" alt="${example.label}"><span>${String(example.identifier).padStart(5, "0")}</span>`;
      button.addEventListener("click", () => {
        resetInput();
        state.activeExample = example.slot;
        button.classList.add("is-active");
        apiRequest({ example_slot: example.slot, example_kind: example.kind });
      });
      elements["examples-grid"].appendChild(button);
    });
    const parameters = new URLSearchParams(window.location.search);
    const requestedExample = Number(parameters.get("example"));
    if (parameters.has("example") && Number.isInteger(requestedExample) && requestedExample >= 0 && requestedExample < data.examples.length) {
      elements["examples-grid"].children[requestedExample].click();
    }
  } catch {
    elements["examples-grid"].textContent = "Held-out examples unavailable.";
  }
}

function clearPrediction() {
  state.result = null;
  state.motionValue = 0;
  state.playing = false;
  state.targetVisible = false;
  if (structureGroup) scene.remove(structureGroup);
  if (targetGroup) scene.remove(targetGroup);
  structureGroup = null;
  targetGroup = null;
  atomPoints = null;
  bondLines = null;
  elements["empty-state"].classList.remove("is-hidden");
  elements["viewer-toolbar"].classList.add("is-disabled");
  elements["motion-controls"].classList.add("is-disabled");
  elements["target-toggle"].classList.add("is-hidden");
  elements["download-button"].disabled = true;
  elements["result-title"].textContent = "No prediction";
  elements["result-subtitle"].textContent = "Load a particle to populate structural diagnostics.";
  elements["metric-list"].innerHTML = `
    <div class="metric-row"><span>Paired RMSD</span><strong>—</strong></div>
    <div class="metric-row"><span>Mean displacement</span><strong>—</strong></div>
    <div class="metric-row"><span>Radius of gyration</span><strong>—</strong></div>`;
  elements["comparison-callout"].classList.add("is-hidden");
  if (fallbackContext && fallbackCanvas) {
    fallbackContext.fillStyle = "#eef3f4";
    fallbackContext.fillRect(0, 0, fallbackCanvas.width, fallbackCanvas.height);
  }
}

function updateSystemCopy(metadata) {
  elements["fact-output"].textContent = `${metadata.atom_count.toLocaleString()} × 3`;
  elements["empty-description"].textContent = `The retained ${metadata.short_name} model returns a ${metadata.representation}.`;
  elements["evidence-stats"].innerHTML = metadata.evidence.map((entry) => (
    `<div class="evidence-stat"><strong>${entry.value}</strong><span>${entry.label}</span></div>`
  )).join("");
  elements["evidence-conclusion"].textContent = metadata.conclusion;
  elements["system-caveat"].textContent = metadata.caveat;
  elements["method-normalization"].textContent = metadata.normalization;
  elements["method-output-title"].textContent = metadata.key === "her2_experimental" ? "Surrogate coordinates" : "Known-target coordinates";
  elements["method-output"].textContent = metadata.representation;
  elements["preprocess-details"].innerHTML = `
    <div><dt>Model input</dt><dd>128 × 128</dd></div>
    <div><dt>Scaling</dt><dd>${metadata.normalization}</dd></div>
    <div><dt>Target status</dt><dd>${metadata.target_status}</dd></div>`;
  updateMethodCopy(metadata);
}

function updateMethodCopy(metadata) {
  const staged = state.methodKey === "staged";
  elements["fact-branch"].textContent = staged
    ? `Staged ${metadata.key === "ak" ? "32D" : "64D"}`
    : "Direct 512D";
  elements["fact-encoder"].textContent = staged ? "Convolutional AE" : "Residual CNN";
  elements["method-title"].textContent = staged
    ? "Staged image-to-coordinate prediction"
    : "Direct image-to-coordinate prediction";
  elements["method-stage-two-title"].textContent = staged ? "Convolutional autoencoder" : "Residual encoder";
  elements["method-stage-two-detail"].textContent = staged
    ? `${metadata.methods[1].description.split(" · ")[0]} image representation`
    : "512-dimensional image representation";
  elements["method-stage-three-title"].textContent = staged ? "Coordinate decoder" : "Coordinate head";
  elements["method-stage-three-detail"].textContent = staged ? "Learned expansion and 1D U-Net" : "Four-layer MLP";
  document.querySelectorAll("[data-method]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.method === state.methodKey);
    button.disabled = !metadata.availability[button.dataset.method];
    button.title = button.disabled ? "Install this method's retained checkpoints to enable inference" : "";
  });
}

function switchMethod(methodKey) {
  const metadata = state.systems.get(state.systemKey);
  if (!metadata || !["direct", "staged"].includes(methodKey)) return;
  state.methodKey = methodKey;
  clearPrediction();
  updateMethodCopy(metadata);
  showToast(`${metadata.short_name} ${methodKey} branch selected.`);
}

async function switchSystem(systemKey, announce = true) {
  const metadata = state.systems.get(systemKey);
  if (!metadata || systemKey === state.systemKey && elements["examples-grid"].children.length) return;
  state.systemKey = systemKey;
  resetInput();
  clearPrediction();
  document.querySelectorAll("[data-system]").forEach((button) => button.classList.toggle("is-active", button.dataset.system === systemKey));
  document.querySelectorAll("[data-summary-system]").forEach((article) => article.classList.toggle("is-active", article.dataset.summarySystem === systemKey));
  updateSystemCopy(metadata);
  elements["examples-grid"].innerHTML = '<span class="examples-loading">Loading…</span>';
  await loadExamples();
  if (announce) showToast(`${metadata.short_name} pipeline selected.`);
}

async function initializeSystems() {
  const response = await fetch("/api/systems");
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "System metadata unavailable.");
  state.systems = new Map(data.systems.map((entry) => [entry.key, entry]));
  const parameters = new URLSearchParams(window.location.search);
  const requested = parameters.get("system");
  const initial = state.systems.has(requested) ? requested : "her2_experimental";
  state.systemKey = "";
  await switchSystem(initial, false);
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    if (!response.ok) throw new Error();
    elements["service-status"].classList.add("is-ready");
    elements["service-status"].innerHTML = `<span class="status-dot"></span>${data.device.toUpperCase()} ready`;
  } catch {
    elements["service-status"].classList.add("is-error");
    elements["service-status"].innerHTML = '<span class="status-dot"></span>Offline';
  }
}

function setPlayIcon(isPlaying) {
  elements["play-button"].innerHTML = `<i data-lucide="${isPlaying ? "pause" : "play"}"></i>`;
  lucide.createIcons();
}

function bindEvents() {
  document.querySelectorAll("[data-system]").forEach((button) => {
    button.addEventListener("click", () => switchSystem(button.dataset.system));
  });
  document.querySelectorAll("[data-summary-system]").forEach((article) => {
    article.addEventListener("click", () => switchSystem(article.dataset.summarySystem));
  });
  document.querySelectorAll("[data-method]").forEach((button) => {
    button.addEventListener("click", () => switchMethod(button.dataset.method));
  });
  elements["drop-zone"].addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      elements["file-input"].click();
    }
  });
  elements["file-input"].addEventListener("change", (event) => selectFile(event.target.files[0]));
  for (const eventName of ["dragenter", "dragover"]) {
    elements["drop-zone"].addEventListener(eventName, (event) => {
      event.preventDefault();
      elements["drop-zone"].classList.add("is-dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    elements["drop-zone"].addEventListener(eventName, (event) => {
      event.preventDefault();
      elements["drop-zone"].classList.remove("is-dragging");
    });
  }
  elements["drop-zone"].addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
  elements["reset-button"].addEventListener("click", resetInput);
  elements["predict-button"].addEventListener("click", async () => {
    if (!state.selectedFile) return;
    try {
      apiRequest(await fileToPayload(state.selectedFile));
    } catch (error) {
      showToast(error.message, true);
    }
  });
  elements["motion-slider"].addEventListener("input", (event) => {
    state.motionValue = Number(event.target.value);
    state.playing = false;
    setPlayIcon(false);
    updateMotionDisplay();
  });
  elements["play-button"].addEventListener("click", () => {
    state.playing = !state.playing;
    setPlayIcon(state.playing);
  });
  document.querySelectorAll(".view-mode").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".view-mode").forEach((item) => item.classList.remove("is-active"));
      button.classList.add("is-active");
      state.playing = false;
      setPlayIcon(false);
      if (button.dataset.mode === "mean") state.motionValue = 0;
      if (button.dataset.mode === "prediction") state.motionValue = 100;
      if (button.dataset.mode === "motion") {
        state.motionValue = 100;
        state.playDirection = -1;
      }
      elements["motion-slider"].value = String(state.motionValue);
      updateMotionDisplay();
      elements["motion-controls"].classList.toggle("is-disabled", button.dataset.mode !== "motion");
    });
  });
  elements["target-toggle"].addEventListener("click", () => {
    state.targetVisible = !state.targetVisible;
    if (targetGroup) targetGroup.visible = state.targetVisible;
    if (!state.webglAvailable) drawFallbackStructure();
    elements["target-toggle"].classList.toggle("is-active", state.targetVisible);
  });
  elements["fit-button"].addEventListener("click", fitStructure);
  elements["fullscreen-button"].addEventListener("click", () => elements["viewer-panel"]?.requestFullscreen?.());
  elements["color-mode"].addEventListener("click", () => {
    state.colorByDisplacement = !state.colorByDisplacement;
    elements["chain-legend"].classList.toggle("is-hidden", state.colorByDisplacement);
    elements["displacement-legend"].classList.toggle("is-hidden", !state.colorByDisplacement);
    elements["legend-title"].textContent = state.colorByDisplacement ? "Displacement" : "Chains";
    elements["color-mode"].textContent = state.colorByDisplacement ? "Color by chain" : "Color by displacement";
    updateMotionDisplay();
  });
  document.querySelectorAll(".result-tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".result-tab").forEach((item) => item.classList.remove("is-active"));
      document.querySelectorAll(".tab-content").forEach((item) => item.classList.remove("is-active"));
      button.classList.add("is-active");
      document.getElementById(`tab-${button.dataset.tab}`).classList.add("is-active");
    });
  });
  elements["download-button"].addEventListener("click", () => {
    if (!state.result?.pdb) return;
    const blob = new Blob([state.result.pdb], { type: "chemical/x-pdb" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${state.result.system_key}_${state.result.source_name.replace(/[^a-z0-9_-]+/gi, "_")}_predicted_coordinates.pdb`;
    link.click();
    URL.revokeObjectURL(link.href);
  });
}

initializeViewer();
bindEvents();
lucide.createIcons();
checkHealth();
initializeSystems().catch((error) => showToast(error.message, true));
