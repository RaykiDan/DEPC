from __future__ import annotations

from PyQt5.QtWidgets import QWidget, QTableWidgetItem
from PyQt5.QtCore import Qt

from ui.setting import Ui_Form
import config


class SettingWindow(QWidget):
    """
    Settings window — model selection, depth range, and annotations.

    Usage
    -----
        # In MainApp.__init__:
        self.setting_window = SettingWindow(on_apply=self._on_settings_applied)
        self.ui.settingButton.clicked.connect(self.setting_window.show)

        # Callback receives a dict with the new settings:
        def _on_settings_applied(self, settings: dict):
            # settings keys: "encoder", "mode", "dmin", "dmax", "annotations"
            ...

    Design notes
    ------------
    - Apply button pushes model + depth range changes immediately.
    - Annotation table edits are also committed on Apply.
    - Close just hides the window — settings already applied are kept.
    - The window reads current values from config on every open (show)
      so it always reflects the live state.
    """

    # ── Encoder display name → internal key used by DepthModel ───────────────
    ENCODER_MAP = {
        "ViT-S": "vits",
        "ViT-B": "vitb",
        "ViT-L": "vitl",
    }
    ENCODER_MAP_INV = {v: k for k, v in ENCODER_MAP.items()}

    MODE_MAP = {
        "Metric":   "metric",
        "Relative": "relative",
    }
    MODE_MAP_INV = {v: k for k, v in MODE_MAP.items()}

    def __init__(self, on_apply, parent=None):
        """
        Parameters
        ----------
        on_apply : callable
            Called with a settings dict when the user presses Apply.
            Signature: on_apply(settings: dict)
        """
        super().__init__(parent)

        self.ui        = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("Settings")

        self._on_apply_cb = on_apply

        self._setup_spinboxes()
        self._populate_table()
        self._connect_buttons()

    # ── Qt show event ─────────────────────────────────────────────────────────

    def showEvent(self, event):
        """Refresh all widgets from current config values every time we open."""
        super().showEvent(event)
        self._apply_setting_overrides()
        self._load_from_config()

    def _apply_setting_overrides(self):
        """
        Override specific widget styles in the settings window so they
        don't inherit the main window's catch-all QLabel border/padding rules.
        """
        # ── settingTitle — keep as bold header ────────────────────────────
        title = getattr(self.ui, "settingTitle", None)
        if title:
            title.setStyleSheet(
                "background-color: transparent;"
                "border: 1px solid #666666;"
                "border-radius: 4px;"
                "font-size: 11pt;"
                "font-weight: bold;"
                "color: #DDDDDD;"
                "padding: 2px;"
            )

        # ── Section sub-headers — plain, non-bold ─────────────────────────
        #   model, depthRange, depthRange_2 (Annotations label)
        subheader_style = (
            "background-color: transparent;"
            "border: none;"
            "font-size: 10pt;"
            "font-weight: normal;"
            "color: #AAAAAA;"
            "padding: 2px;"
        )
        for name in ("model", "depthRange", "depthRange_2"):
            lbl = getattr(self.ui, name, None)
            if lbl:
                lbl.setStyleSheet(subheader_style)

        # ── Field labels — plain ──────────────────────────────────────────
        field_style = (
            "background-color: transparent;"
            "border: none;"
            "font-size: 10pt;"
            "color: #CCCCCC;"
            "padding: 2px;"
        )
        for name in ("encoder", "mode", "dmin", "dmax"):
            lbl = getattr(self.ui, name, None)
            if lbl:
                lbl.setStyleSheet(field_style)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_spinboxes(self):
        """Configure spinbox ranges and step sizes."""
        for box in (self.ui.dminBox, self.ui.dmaxBox):
            box.setMinimum(0.01)
            box.setMaximum(20.0)
            box.setSingleStep(0.1)
            box.setDecimals(2)

    def _populate_table(self):
        """Set up annotation table columns and load current annotations."""
        self.ui.tableWidget.setColumnCount(3)
        self.ui.tableWidget.setHorizontalHeaderLabels(["Name", "Near", "Far"])
        self.ui.tableWidget.horizontalHeader().setStretchLastSection(True)
        self._load_annotations_to_table(config.ANNOTATIONS)

    def _connect_buttons(self):
        self.ui.applyDepthButton.clicked.connect(self._on_apply)
        self.ui.addAnnotationButton.clicked.connect(self._add_row)
        self.ui.removeAnnotationButton.clicked.connect(self._remove_row)
        self.ui.closeButton.clicked.connect(self.hide)

    # ── Load config → UI ──────────────────────────────────────────────────────

    def _load_from_config(self):
        """Sync all widgets to the current in-memory config values."""
        # Encoder
        enc_display = self.ENCODER_MAP_INV.get(config.CURRENT_ENCODER, "ViT-S")
        idx = self.ui.encoderBox.findText(enc_display)
        if idx >= 0:
            self.ui.encoderBox.setCurrentIndex(idx)

        # Mode
        mode_display = self.MODE_MAP_INV.get(config.CURRENT_MODE, "Metric")
        idx = self.ui.modeBox.findText(mode_display)
        if idx >= 0:
            self.ui.modeBox.setCurrentIndex(idx)

        # Depth range
        self.ui.dminBox.setValue(config.DMIN)
        self.ui.dmaxBox.setValue(config.DMAX)

        # Annotations
        self._load_annotations_to_table(config.ANNOTATIONS)

    # ── Apply ─────────────────────────────────────────────────────────────────

    def _on_apply(self):
        """Read all widgets, update config in memory, call the apply callback."""

        # ── Model ─────────────────────────────────────────────────────────
        encoder = self.ENCODER_MAP.get(self.ui.encoderBox.currentText(), "vits")
        mode    = self.MODE_MAP.get(self.ui.modeBox.currentText(), "metric")

        # ── Depth range ───────────────────────────────────────────────────
        dmin = self.ui.dminBox.value()
        dmax = self.ui.dmaxBox.value()

        if dmin >= dmax:
            print("[SettingWindow] dmin must be less than dmax — not applied.")
            return

        # ── Annotations ───────────────────────────────────────────────────
        annotations = self._read_annotations_from_table()

        # ── Update config in memory ───────────────────────────────────────
        config.CURRENT_ENCODER = encoder
        config.CURRENT_MODE    = mode
        config.DMIN            = dmin
        config.DMAX            = dmax
        config.ANNOTATIONS     = annotations

        print(f"[SettingWindow] Applied — encoder={encoder}, mode={mode}, "
              f"dmin={dmin}, dmax={dmax}, annotations={len(annotations)}")

        # ── Notify MainApp ────────────────────────────────────────────────
        self._on_apply_cb({
            "encoder":     encoder,
            "mode":        mode,
            "dmin":        dmin,
            "dmax":        dmax,
            "annotations": annotations,
        })

    # ── Annotation table helpers ──────────────────────────────────────────────

    def _load_annotations_to_table(self, annotations: list):
        """Populate the table from a list of annotation dicts."""
        self.ui.tableWidget.setRowCount(0)
        for ann in annotations:
            self._append_row(
                name=ann.get("name", ""),
                near=ann.get("depth_min", 0.0),
                far=ann.get("depth_max", 0.0),
            )

    def _read_annotations_from_table(self) -> list:
        """
        Read the current table contents into a list of annotation dicts.
        Skips rows with empty names or invalid depth values.
        """
        annotations = []
        for row in range(self.ui.tableWidget.rowCount()):
            name_item = self.ui.tableWidget.item(row, 0)
            near_item = self.ui.tableWidget.item(row, 1)
            far_item  = self.ui.tableWidget.item(row, 2)

            if not name_item or not name_item.text().strip():
                continue

            try:
                near = float(near_item.text()) if near_item else 0.0
                far  = float(far_item.text())  if far_item  else 0.0
            except ValueError:
                continue

            if near >= far:
                continue

            annotations.append({
                "name":      name_item.text().strip(),
                "depth_min": near,
                "depth_max": far,
                "color":     (0, 0, 0),
            })
        return annotations

    def _append_row(self, name: str = "", near: float = 0.0, far: float = 0.0):
        """Add one row to the annotation table."""
        row = self.ui.tableWidget.rowCount()
        self.ui.tableWidget.insertRow(row)
        self.ui.tableWidget.setItem(row, 0, QTableWidgetItem(name))
        self.ui.tableWidget.setItem(row, 1, QTableWidgetItem(str(near)))
        self.ui.tableWidget.setItem(row, 2, QTableWidgetItem(str(far)))

    def _add_row(self):
        """Add a blank row for the user to fill in."""
        self._append_row()

    def _remove_row(self):
        """Remove the currently selected row."""
        row = self.ui.tableWidget.currentRow()
        if row >= 0:
            self.ui.tableWidget.removeRow(row)