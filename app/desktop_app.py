#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import sys
import traceback
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from inference import PredictionService, decode_image


APP_DIR = Path(__file__).resolve().parent
SYSTEM_ORDER = ["ak", "her2_synthetic", "her2_experimental"]
SYSTEM_LABELS = {
    "ak": "AK",
    "her2_synthetic": "Synthetic HER2",
    "her2_experimental": "Experimental HER2",
}
METHOD_ORDER = ["direct", "staged"]
METHOD_LABELS = {
    "direct": "Direct · residual CNN",
    "staged": "Staged · CAE → 1D U-Net",
}


class WorkerSignals(QObject):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)


class InferenceJob(QRunnable):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.signals.completed.emit(self.operation())
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class ParticlePreview(QLabel):
    def __init__(self):
        super().__init__("No image selected")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(220, 220)
        self.setObjectName("particlePreview")

    def show_array(self, image: np.ndarray):
        array = np.asarray(image, dtype=np.float32)
        low, high = np.percentile(array, [1, 99])
        scaled = np.clip((array - low) / max(float(high - low), 1e-6), 0, 1)
        pixels = np.ascontiguousarray(np.uint8(scaled * 255))
        qimage = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format_Grayscale8).copy()
        self.setPixmap(QPixmap.fromImage(qimage).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class StructureCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(figsize=(8, 7), facecolor="#101820")
        self.axes = self.figure.add_subplot(111, projection="3d")
        super().__init__(self.figure)
        self.setMinimumSize(580, 520)
        self.result = None
        self.motion = 1.0
        self.show_target = False
        self._empty()

    def _style_axes(self):
        self.axes.set_facecolor("#101820")
        self.axes.set_axis_off()
        self.axes.grid(False)
        self.axes.view_init(elev=17, azim=-62)

    def _empty(self):
        self.axes.clear()
        self._style_axes()
        self.axes.text2D(
            0.5,
            0.5,
            "Select a particle image or reference control",
            transform=self.axes.transAxes,
            ha="center",
            va="center",
            color="#92a4b4",
            fontsize=12,
        )
        self.draw_idle()

    def set_result(self, result: dict, motion: float = 1.0, show_target: bool = False):
        self.result = result
        self.motion = motion
        self.show_target = show_target
        self.redraw()

    def redraw(self):
        if not self.result:
            self._empty()
            return
        prediction = np.asarray(self.result["prediction"], dtype=np.float32)
        mean = np.asarray(self.result["training_mean"], dtype=np.float32)
        coordinates = mean + self.motion * (prediction - mean)
        target = self.result.get("paired_target")
        self.axes.clear()
        self._style_axes()

        if self.show_target and target is not None:
            self._draw_structure(np.asarray(target), "#d3dae1", 0.8, 0.45, dashed=True)
        self._draw_structure(coordinates, None, 1.45, 0.95)
        self._equalize(coordinates)
        self.axes.text2D(
            0.025,
            0.965,
            f"{self.result['model']['short_name']}  |  "
            f"{self.result['model']['method_key'].capitalize()}  |  {len(coordinates):,} positions",
            transform=self.axes.transAxes,
            color="#edf5fa",
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        self.draw_idle()

    def _draw_structure(self, coordinates, fallback_color, width, alpha, dashed=False):
        colors = self.result.get("chain_colors", {})
        atoms = self.result["topology"]["atoms"]
        for segment in self.result["segments"]:
            indices = np.asarray(segment, dtype=int)
            indices = indices[(indices >= 0) & (indices < len(coordinates))]
            if len(indices) < 2:
                continue
            chain = str(atoms[int(indices[0])].get("chain", "A"))
            color = fallback_color or colors.get(chain, "#2f80ed")
            points = coordinates[indices]
            self.axes.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=color,
                linewidth=width,
                alpha=alpha,
                linestyle="--" if dashed else "-",
            )
        sample = coordinates[:: max(1, len(coordinates) // 550)]
        self.axes.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=4, c=fallback_color or "#57a6ff", alpha=alpha)

    def _equalize(self, coordinates):
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        center = (minimum + maximum) / 2
        radius = max(float((maximum - minimum).max()) * 0.58, 1.0)
        self.axes.set_xlim(center[0] - radius, center[0] + radius)
        self.axes.set_ylim(center[1] - radius, center[1] + radius)
        self.axes.set_zlim(center[2] - radius, center[2] + radius)
        self.axes.set_box_aspect((1, 1, 1))


class MetricCard(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 11, 14, 11)
        self.value = QLabel("--")
        self.value.setObjectName("metricValue")
        label = QLabel(title)
        label.setObjectName("metricLabel")
        layout.addWidget(self.value)
        layout.addWidget(label)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Conformational Coordinate Lab")
        self.resize(1500, 930)
        self.setMinimumSize(1120, 760)
        self.service = PredictionService()
        self.thread_pool = QThreadPool.globalInstance()
        self.selected_path: Path | None = None
        self.result: dict | None = None
        self.animation_direction = -1
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_animation)
        self._build_ui()
        self._apply_style()
        self.system_combo.setCurrentIndex(2)

    @property
    def system_key(self):
        return SYSTEM_ORDER[self.system_combo.currentIndex()]

    @property
    def method_key(self):
        return METHOD_ORDER[self.method_combo.currentIndex()]

    def _build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 14, 24, 14)
        title_box = QVBoxLayout()
        title = QLabel("Conformational Coordinate Lab")
        title.setObjectName("appTitle")
        subtitle = QLabel("Local image-conditioned structural prediction")
        subtitle.setObjectName("appSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box)
        header_layout.addStretch()
        self.system_combo = QComboBox()
        self.system_combo.addItems([SYSTEM_LABELS[key] for key in SYSTEM_ORDER])
        self.system_combo.currentIndexChanged.connect(self.change_system)
        header_layout.addWidget(self.system_combo)
        self.method_combo = QComboBox()
        self.method_combo.addItems([METHOD_LABELS[key] for key in METHOD_ORDER])
        self.method_combo.currentIndexChanged.connect(self.change_method)
        header_layout.addWidget(self.method_combo)
        device = QLabel(f"●  {str(self.service.device).upper()} ready")
        device.setObjectName("deviceStatus")
        header_layout.addWidget(device)
        root_layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([310, 840, 335])
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready. Images and model outputs remain on this computer.")

    def _build_left_panel(self):
        panel = QFrame()
        panel.setObjectName("sidePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        heading = QLabel("PARTICLE INPUT")
        heading.setObjectName("eyebrow")
        layout.addWidget(heading)
        self.preview = ParticlePreview()
        layout.addWidget(self.preview)
        self.input_name = QLabel("No file selected")
        self.input_name.setWordWrap(True)
        layout.addWidget(self.input_name)
        choose = QPushButton("Choose image…")
        choose.setObjectName("secondaryButton")
        choose.clicked.connect(self.choose_file)
        layout.addWidget(choose)
        self.predict_button = QPushButton("Predict coordinates")
        self.predict_button.setObjectName("primaryButton")
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(self.predict_upload)
        layout.addWidget(self.predict_button)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        layout.addWidget(divider)
        layout.addWidget(QLabel("REFERENCE CONTROLS", objectName="eyebrow"))
        self.examples = QListWidget()
        self.examples.itemDoubleClicked.connect(self.predict_example)
        layout.addWidget(self.examples, 1)
        example_button = QPushButton("Run selected control")
        example_button.clicked.connect(lambda: self.predict_example(self.examples.currentItem()))
        layout.addWidget(example_button)
        return panel

    def _build_center_panel(self):
        panel = QFrame()
        panel.setObjectName("viewerPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 12)
        self.canvas = StructureCanvas()
        layout.addWidget(self.canvas, 1)
        controls = QHBoxLayout()
        self.play_button = QToolButton()
        self.play_button.setText("▶")
        self.play_button.setToolTip("Play interpolation")
        self.play_button.clicked.connect(self.toggle_animation)
        controls.addWidget(self.play_button)
        controls.addWidget(QLabel("Training mean"))
        self.motion_slider = QSlider(Qt.Horizontal)
        self.motion_slider.setRange(0, 100)
        self.motion_slider.setValue(100)
        self.motion_slider.valueChanged.connect(self.update_motion)
        controls.addWidget(self.motion_slider, 1)
        controls.addWidget(QLabel("Prediction"))
        self.target_toggle = QCheckBox("Paired target overlay")
        self.target_toggle.setEnabled(False)
        self.target_toggle.toggled.connect(self.update_motion)
        controls.addWidget(self.target_toggle)
        fit = QPushButton("Fit view")
        fit.clicked.connect(self.update_motion)
        controls.addWidget(fit)
        layout.addLayout(controls)
        note = QLabel("Displayed motion is linear coordinate interpolation, not molecular dynamics.")
        note.setObjectName("caveat")
        note.setAlignment(Qt.AlignCenter)
        layout.addWidget(note)
        return panel

    def _build_right_panel(self):
        panel = QFrame()
        panel.setObjectName("sidePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        layout.addWidget(QLabel("PREDICTION", objectName="eyebrow"))
        self.result_title = QLabel("No prediction")
        self.result_title.setObjectName("resultTitle")
        self.result_title.setWordWrap(True)
        layout.addWidget(self.result_title)
        self.model_description = QLabel()
        self.model_description.setWordWrap(True)
        layout.addWidget(self.model_description)

        cards = QGridLayout()
        self.rmsd_card = MetricCard("paired-target raw RMSD")
        self.mean_card = MetricCard("distance from training mean")
        self.rg_card = MetricCard("prediction radius of gyration")
        self.positions_card = MetricCard("coordinate positions")
        for index, card in enumerate([self.rmsd_card, self.mean_card, self.rg_card, self.positions_card]):
            cards.addWidget(card, index // 2, index % 2)
        layout.addLayout(cards)

        layout.addWidget(QLabel("PREPROCESSING", objectName="eyebrow"))
        self.preprocessing = QLabel("Select an input to inspect preprocessing.")
        self.preprocessing.setWordWrap(True)
        self.preprocessing.setObjectName("infoText")
        layout.addWidget(self.preprocessing)
        layout.addWidget(QLabel("INTERPRETATION", objectName="eyebrow"))
        self.interpretation = QLabel()
        self.interpretation.setWordWrap(True)
        self.interpretation.setObjectName("infoText")
        layout.addWidget(self.interpretation)
        layout.addStretch()
        self.export_button = QPushButton("Export predicted PDB…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_pdb)
        layout.addWidget(self.export_button)
        return panel

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f4f7f9; color: #15232d; font-family: Inter, Arial; font-size: 13px; }
            #header { background: #ffffff; border-bottom: 1px solid #d8e1e7; }
            #appTitle { font-size: 22px; font-weight: 700; color: #102a3a; }
            #appSubtitle { color: #5e7180; }
            #deviceStatus { color: #18794e; font-weight: 600; padding-left: 12px; }
            #sidePanel { background: #ffffff; border-right: 1px solid #d8e1e7; }
            #viewerPanel { background: #101820; }
            #eyebrow { color: #607b8b; font-size: 11px; font-weight: 700; }
            #particlePreview { background: #17242d; color: #8ea2af; border: 1px solid #263a47; }
            #primaryButton { background: #146cb7; color: white; border: 0; padding: 11px; font-weight: 700; }
            #primaryButton:disabled { background: #aebbc4; }
            #secondaryButton, QPushButton, QToolButton { background: #eef3f6; border: 1px solid #c7d3da; padding: 8px; }
            QPushButton:hover, QToolButton:hover { background: #e0ebf1; }
            QComboBox { background: white; border: 1px solid #bdcad2; padding: 7px 12px; min-width: 160px; }
            QListWidget { background: #f8fafb; border: 1px solid #d5dfe5; outline: 0; }
            QListWidget::item { padding: 9px; border-bottom: 1px solid #e4eaee; }
            QListWidget::item:selected { background: #dceeff; color: #0b5795; }
            #resultTitle { font-size: 19px; font-weight: 700; color: #102a3a; }
            #metricCard { background: #f4f8fa; border: 1px solid #d5e0e6; }
            #metricValue { font-size: 17px; font-weight: 700; color: #0d609f; }
            #metricLabel { font-size: 10px; color: #647684; }
            #infoText { background: #f6f8f9; border-left: 3px solid #75a8c9; padding: 10px; color: #3c5261; }
            #caveat { color: #aebdca; font-style: italic; padding: 4px; }
            QSlider::groove:horizontal { height: 4px; background: #435563; }
            QSlider::handle:horizontal { background: #3b9cff; width: 16px; margin: -6px 0; border-radius: 8px; }
            QProgressBar { border: 0; background: #dbe5ea; height: 3px; }
            """
        )

    def change_system(self, index):
        if index < 0:
            return
        service = self.service.system(self.system_key)
        config = service.config
        self.selected_path = None
        self.result = None
        self.predict_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.target_toggle.setChecked(False)
        self.target_toggle.setEnabled(False)
        self.input_name.setText("No file selected")
        self.preview.clear()
        self.preview.setText("No image selected")
        self.result_title.setText("No prediction")
        self._update_model_description()
        self.interpretation.setText(f"{config.conclusion}\n\n{config.caveat}")
        self.preprocessing.setText(f"Uploaded images: {config.normalization}.")
        self.canvas.result = None
        self.canvas._empty()
        self.examples.clear()
        for entry in service.example_catalog():
            item = QListWidgetItem(entry["label"])
            item.setData(Qt.UserRole, (entry["kind"], entry["slot"]))
            self.examples.addItem(item)
        if self.examples.count():
            self.examples.setCurrentRow(0)
        self.statusBar().showMessage(f"{config.short_name} selected. {config.target_status}.")

    def change_method(self, index):
        if index < 0 or not hasattr(self, "canvas"):
            return
        self.result = None
        self.export_button.setEnabled(False)
        self.target_toggle.setChecked(False)
        self.target_toggle.setEnabled(False)
        self.result_title.setText("No prediction")
        self.canvas.result = None
        self.canvas._empty()
        self._update_model_description()
        self.statusBar().showMessage(
            f"{METHOD_LABELS[self.method_key]} selected for {SYSTEM_LABELS[self.system_key]}."
        )

    def _update_model_description(self):
        service = self.service.system(self.system_key)
        config = service.config
        if self.method_key == "direct":
            method = "Direct residual CNN · 512D coordinate head"
        else:
            method = f"Staged CAE · {config.staged_latent_dim}D latent · 1D U-Net decoder"
        self.model_description.setText(f"{config.full_name}\n{method} · {config.atom_count:,} × 3 output")

    def choose_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Choose particle image",
            str(Path.home()),
            "Particle images (*.spi *.png *.tif *.tiff *.jpg *.jpeg *.npy)",
        )
        if not filename:
            return
        path = Path(filename)
        try:
            content = base64.b64encode(path.read_bytes()).decode("ascii")
            image, details = decode_image(path.name, content, self.service.system(self.system_key).config)
        except Exception as error:
            QMessageBox.critical(self, "Unreadable image", str(error))
            return
        self.selected_path = path
        self.preview.show_array(image)
        self.input_name.setText(f"{path.name}\n{path.stat().st_size / 1024:.1f} KB")
        warning = details.get("shape_warning") or "No resizing or cropping was required."
        self.preprocessing.setText(
            f"Input {details['original_shape'][0]} × {details['original_shape'][1]} → model 128 × 128\n"
            f"{details['normalization']}\n{warning}"
        )
        self.predict_button.setEnabled(True)

    def predict_upload(self):
        if not self.selected_path:
            return
        path = self.selected_path
        key = self.system_key
        method = self.method_key
        content = base64.b64encode(path.read_bytes()).decode("ascii")
        self.run_job(lambda: self.service.system(key).predict_upload(path.name, content, method))

    def predict_example(self, item):
        if item is None:
            return
        kind, slot = item.data(Qt.UserRole)
        slot = int(slot)
        key = self.system_key
        method = self.method_key
        system_service = self.service.system(key)
        examples = system_service.examples if kind == "held_out" else system_service.fresh_examples
        example_image = np.asarray(examples["images"][slot])
        self.preview.show_array(example_image)
        self.input_name.setText(item.text())
        self.run_job(lambda: self.service.system(key).predict_example(slot, method, kind))

    def run_job(self, operation):
        self.predict_button.setEnabled(False)
        self.statusBar().showMessage("Running GPU inference…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        job = InferenceJob(operation)
        job.signals.completed.connect(self.apply_result)
        job.signals.failed.connect(self.inference_failed)
        self.thread_pool.start(job)

    def apply_result(self, result):
        QApplication.restoreOverrideCursor()
        self.result = result
        self.predict_button.setEnabled(self.selected_path is not None)
        self.export_button.setEnabled(True)
        self.motion_slider.setValue(100)
        self.target_toggle.setChecked(False)
        self.target_toggle.setEnabled(result.get("paired_target") is not None)
        self.canvas.set_result(result)
        self.result_title.setText(
            f"{result['source_name']} · {result['model']['method_key'].capitalize()}"
        )
        self.model_description.setText(
            f"{result['model']['full_name']}\n{result['model']['branch']} · "
            f"{len(result['prediction']):,} × 3 output"
        )
        metrics = result["metrics"]
        paired = metrics.get("paired_target_raw_rmsd")
        self.rmsd_card.value.setText("n/a" if paired is None else f"{paired:.3f} Å")
        self.mean_card.value.setText(f"{metrics['displacement_from_training_mean_rmsd']:.3f} Å")
        self.rg_card.value.setText(f"{metrics['prediction_radius_of_gyration']:.2f} Å")
        self.positions_card.value.setText(f"{len(result['prediction']):,}")
        preprocessing = result["preprocessing"]
        self.preprocessing.setText(
            f"{preprocessing.get('normalization', 'stored normalized input')}\n"
            f"Original shape: {' × '.join(map(str, preprocessing.get('original_shape', [])))}\n"
            f"{preprocessing.get('shape_warning') or preprocessing.get('source', 'No resizing required.')}"
        )
        self.statusBar().showMessage("Prediction complete. Output remains local.")

    def inference_failed(self, details):
        QApplication.restoreOverrideCursor()
        self.predict_button.setEnabled(self.selected_path is not None)
        self.statusBar().showMessage("Inference failed.")
        QMessageBox.critical(self, "Inference failed", details.splitlines()[-1])

    def update_motion(self):
        if self.result:
            self.canvas.set_result(self.result, self.motion_slider.value() / 100, self.target_toggle.isChecked())

    def toggle_animation(self):
        if not self.result:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("▶")
        else:
            self.timer.start(45)
            self.play_button.setText("❚❚")

    def advance_animation(self):
        value = self.motion_slider.value() + self.animation_direction * 2
        if value <= 0:
            value = 0
            self.animation_direction = 1
        elif value >= 100:
            value = 100
            self.animation_direction = -1
        self.motion_slider.setValue(value)

    def export_pdb(self):
        if not self.result:
            return
        stem = Path(self.result["source_name"]).stem
        filename, _ = QFileDialog.getSaveFileName(self, "Export predicted PDB", f"{stem}_prediction.pdb", "PDB (*.pdb)")
        if filename:
            Path(filename).write_text(self.result["pdb"])
            self.statusBar().showMessage(f"Saved {filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Local desktop interface for three image-to-coordinate models")
    parser.add_argument("--smoke-test", action="store_true", help="Run one held-out inference without opening a window")
    parser.add_argument("--system", choices=SYSTEM_ORDER, default="her2_experimental")
    parser.add_argument("--method", choices=METHOD_ORDER, default="direct")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke_test:
        service = PredictionService()
        result = service.system(args.system).predict_example(0, args.method)
        print(
            f"desktop inference smoke passed: system={args.system}, method={args.method}, device={service.device}, "
            f"positions={len(result['prediction'])}, source={result['source_name']}"
        )
        return 0
    application = QApplication(sys.argv)
    application.setApplicationName("Conformational Coordinate Lab")
    application.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return application.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
