from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)

from saxshell.mdtrajectory.frame.cp2k_ener import CP2KEnergyData
from saxshell.mdtrajectory.frame.cutoff_analysis import CP2KEnergyAnalyzer


class EnergyFigureCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas with draggable vertical cutoff line."""

    cutoff_changed = Signal(float)

    def __init__(self) -> None:
        self.figure = Figure(figsize=(7, 4.8), dpi=100)
        super().__init__(self.figure)

        self.ax1 = self.figure.add_subplot(311)
        self.ax2 = self.figure.add_subplot(312, sharex=self.ax1)
        self.ax3 = self.figure.add_subplot(313, sharex=self.ax1)

        self._time_fs = None
        self._cutoff_lines = []
        self._dragging = False
        self.cutoff_x_fs: float | None = None

        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

    def plot_energy(
        self,
        energy_data: CP2KEnergyData,
        suggested_fs: float | None,
        temp_target_k: float | None,
        selected_fs: float | None = None,
    ) -> None:
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        time_fs = energy_data.time_fs
        self._time_fs = time_fs

        self.ax1.plot(time_fs, energy_data.kinetic, color="red")
        self.ax2.plot(time_fs, energy_data.temperature, color="blue")
        self.ax3.plot(time_fs, energy_data.potential, color="green")

        self.ax1.set_ylabel("Kinetic")
        self.ax2.set_ylabel("Temp (K)")
        self.ax3.set_ylabel("Potential")
        self.ax3.set_xlabel("Time (fs)")

        if temp_target_k is not None:
            self.ax2.axhline(
                temp_target_k,
                color="gray",
                linestyle="-",
                label=f"Target = {temp_target_k:.1f} K",
            )

        if suggested_fs is not None:
            for ax in (self.ax1, self.ax2, self.ax3):
                ax.axvline(
                    suggested_fs,
                    color="black",
                    linestyle="--",
                    label=f"Suggested = {suggested_fs:.2f} fs",
                )

        if selected_fs is not None:
            self.cutoff_x_fs = float(selected_fs)
        elif suggested_fs is not None:
            self.cutoff_x_fs = float(suggested_fs)
        elif len(time_fs) > 0:
            self.cutoff_x_fs = float(time_fs[len(time_fs) // 2])
        else:
            self.cutoff_x_fs = None

        self._cutoff_lines = []
        if self.cutoff_x_fs is not None:
            self._cutoff_lines = [
                ax.axvline(
                    self.cutoff_x_fs,
                    color="magenta",
                    linestyle="--",
                    linewidth=2,
                )
                for ax in (self.ax1, self.ax2, self.ax3)
            ]

        for ax in (self.ax1, self.ax2, self.ax3):
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper right")
        self.figure.tight_layout(pad=2.0)
        self.draw_idle()

    def clear_plot(self) -> None:
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax1.set_ylabel("Kinetic")
        self.ax2.set_ylabel("Temp (K)")
        self.ax3.set_ylabel("Potential")
        self.ax3.set_xlabel("Time (fs)")
        self.ax2.text(
            0.5,
            0.5,
            "Load a CP2K .ener file to view the energy profile.",
            transform=self.ax2.transAxes,
            ha="center",
            va="center",
        )
        self._time_fs = None
        self._cutoff_lines = []
        self.cutoff_x_fs = None
        self.figure.tight_layout(pad=2.0)
        self.draw_idle()

    def set_cutoff(self, cutoff_fs: float | None) -> None:
        self.cutoff_x_fs = cutoff_fs
        if not self._cutoff_lines:
            return

        visible = cutoff_fs is not None
        for line in self._cutoff_lines:
            line.set_visible(visible)

        if cutoff_fs is None:
            self.draw_idle()
            return

        if self._time_fs is not None and len(self._time_fs) > 0:
            x_min = float(self._time_fs.min())
            x_max = float(self._time_fs.max())
            cutoff_fs = min(max(float(cutoff_fs), x_min), x_max)
            self.cutoff_x_fs = cutoff_fs

        for line in self._cutoff_lines:
            line.set_xdata([cutoff_fs, cutoff_fs])
        self.draw_idle()

    def _on_press(self, event) -> None:
        if event.xdata is None or self._time_fs is None:
            return

        x_range = float(self._time_fs.max() - self._time_fs.min())
        tol = max(0.01 * x_range, 1.0e-12)

        for line in self._cutoff_lines:
            x0 = float(line.get_xdata()[0])
            if abs(float(event.xdata) - x0) < tol:
                self._dragging = True
                break

    def _on_motion(self, event) -> None:
        if not self._dragging or event.xdata is None:
            return

        self.cutoff_x_fs = float(event.xdata)
        for line in self._cutoff_lines:
            line.set_xdata([self.cutoff_x_fs, self.cutoff_x_fs])

        self.cutoff_changed.emit(self.cutoff_x_fs)
        self.draw_idle()

    def _on_release(self, event) -> None:
        self._dragging = False
        if event.xdata is not None:
            self.cutoff_x_fs = float(event.xdata)
            self.cutoff_changed.emit(self.cutoff_x_fs)


class CutoffPanel(QGroupBox):
    """Panel for CP2K .ener loading and cutoff selection."""

    selection_changed = Signal()
    suggestion_updated = Signal(float)
    cutoff_selected = Signal(float)

    def __init__(self) -> None:
        super().__init__("Cutoff Selection")
        self.energy_data: CP2KEnergyData | None = None
        self._suggested_cutoff_fs: float | None = None
        self._build_ui()
        self.reset()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        form = QFormLayout()

        self.temp_target_spin = QDoubleSpinBox()
        self.temp_target_spin.setRange(0.0, 5000.0)
        self.temp_target_spin.setDecimals(2)
        self.temp_target_spin.setValue(300.0)
        self.temp_target_spin.setToolTip(
            "Temperature setpoint in kelvin. A horizontal guide line is drawn "
            "at this value on the temperature plot."
        )

        self.temp_tol_spin = QDoubleSpinBox()
        self.temp_tol_spin.setRange(0.0, 1000.0)
        self.temp_tol_spin.setDecimals(2)
        self.temp_tol_spin.setValue(1.0)
        self.temp_tol_spin.setToolTip(
            "Allowed deviation from the target temperature when suggesting a "
            "steady-state cutoff."
        )

        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 10**6)
        self.window_spin.setValue(3)
        self.window_spin.setToolTip(
            "Number of consecutive energy samples that must satisfy the "
            "steady-state criteria."
        )

        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(-1.0, 10**12)
        self.cutoff_spin.setDecimals(3)
        self.cutoff_spin.setSingleStep(10.0)
        self.cutoff_spin.setKeyboardTracking(False)
        self.cutoff_spin.setSpecialValueText("None")
        self.cutoff_spin.setValue(-1.0)
        self.cutoff_spin.setToolTip(
            "Manual export cutoff in femtoseconds. Frames before this time "
            "will be excluded when cutoff export is enabled."
        )

        form.addRow("Target Temp (K)", self.temp_target_spin)
        form.addRow("Tolerance (K)", self.temp_tol_spin)
        form.addRow("Window", self.window_spin)
        form.addRow("Enforced cutoff (fs)", self.cutoff_spin)

        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.suggest_button = QPushButton("Suggest Cutoff")
        self.clear_button = QPushButton("Clear Cutoff")
        self.suggest_button.setToolTip(
            "Analyze the loaded .ener file and suggest the first steady-state "
            "time that matches the target temperature."
        )
        self.clear_button.setToolTip(
            "Remove the currently selected cutoff so all frames remain "
            "eligible for export."
        )
        button_row.addWidget(self.suggest_button)
        button_row.addWidget(self.clear_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.canvas = EnergyFigureCanvas()
        self.canvas.setToolTip(
            "Drag the magenta vertical line to choose a manual cutoff time."
        )
        self.canvas.cutoff_changed.connect(self._sync_cutoff_from_plot)
        self.plot_toolbar = NavigationToolbar(self.canvas, self)
        self.plot_toolbar.setToolTip(
            "Matplotlib navigation toolbar for zoom, pan, and saving the plot."
        )
        layout.addWidget(self.plot_toolbar)
        layout.addWidget(self.canvas)

        self.cutoff_label = QLabel("Selected cutoff: None")
        self.cutoff_label.setToolTip(
            "Current cutoff that will be used when export cutoff is enabled."
        )
        layout.addWidget(self.cutoff_label)

        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMinimumHeight(90)
        self.summary_box.setToolTip(
            "Summary of the loaded .ener file and any suggested cutoff."
        )
        layout.addWidget(self.summary_box)

        self.suggest_button.clicked.connect(
            lambda _checked=False: self.suggest_cutoff()
        )
        self.clear_button.clicked.connect(
            lambda _checked=False: self.clear_cutoff()
        )
        self.temp_target_spin.valueChanged.connect(
            lambda _value: self._refresh_energy_plot()
        )
        self.cutoff_spin.valueChanged.connect(self._on_cutoff_spin_changed)

        self.setLayout(layout)

    def reset(self) -> None:
        self.energy_data = None
        self._suggested_cutoff_fs = None
        self.suggest_button.setEnabled(False)
        self.canvas.clear_plot()
        self._set_cutoff_value(None, emit_signals=False)
        self.summary_box.setPlainText(
            "Load a CP2K .ener file to plot the energy profile.\n"
            "You can still enter a manual cutoff time."
        )

    def load_energy_data(self, energy_data: CP2KEnergyData) -> None:
        self.energy_data = energy_data
        self.suggest_button.setEnabled(True)
        self._refresh_energy_plot()
        self.summary_box.setPlainText(
            f"Loaded energy file: {energy_data.filepath}\n"
            f"Time range: {energy_data.time_min_fs:.3f} fs to "
            f"{energy_data.time_max_fs:.3f} fs\n"
            f"Samples: {energy_data.n_points}"
        )
        self.selection_changed.emit()

    def suggest_cutoff(self) -> None:
        if self.energy_data is None:
            raise ValueError("No CP2K .ener data loaded.")

        analyzer = CP2KEnergyAnalyzer(self.energy_data)
        result = analyzer.suggest_steady_state_cutoff(
            temp_target_k=self.temp_target_spin.value(),
            temp_tol_k=self.temp_tol_spin.value(),
            window=self.window_spin.value(),
        )

        self._suggested_cutoff_fs = result.cutoff_time_fs
        self._refresh_energy_plot(
            selected_fs=(
                result.cutoff_time_fs
                if result.cutoff_time_fs is not None
                else self.get_selected_cutoff()
            )
        )

        if result.cutoff_time_fs is None:
            self.summary_box.setPlainText(
                "No steady-state cutoff was detected.\n"
                "Enter a manual cutoff if you still want to export from a "
                "specific time."
            )
            self.selection_changed.emit()
            return

        self._set_cutoff_value(result.cutoff_time_fs)
        self.summary_box.setPlainText(
            "Suggested cutoff: "
            f"{result.cutoff_time_fs:.3f} fs\n"
            f"Target temperature: {result.temp_target_k:.2f} K\n"
            f"Tolerance: {result.temp_tol_k:.2f} K\n"
            f"Window: {result.window}"
        )
        self.suggestion_updated.emit(result.cutoff_time_fs)

    def clear_cutoff(self) -> None:
        self._set_cutoff_value(None)

    def get_selected_cutoff(self) -> float | None:
        value = self.cutoff_spin.value()
        return None if value < 0.0 else value

    def get_suggested_cutoff(self) -> float | None:
        return self._suggested_cutoff_fs

    def _sync_cutoff_from_plot(self, value: float) -> None:
        self._set_cutoff_value(value, sync_canvas=False)

    def _on_cutoff_spin_changed(self, value: float) -> None:
        self._set_cutoff_value(value)

    def _set_cutoff_value(
        self,
        value: float | None,
        *,
        emit_signals: bool = True,
        sync_canvas: bool = True,
    ) -> None:
        cutoff_fs = None if value is None or value < 0.0 else float(value)

        self.cutoff_spin.blockSignals(True)
        self.cutoff_spin.setValue(-1.0 if cutoff_fs is None else cutoff_fs)
        self.cutoff_spin.blockSignals(False)

        if sync_canvas:
            self.canvas.set_cutoff(cutoff_fs)

        self._update_cutoff_label(cutoff_fs)

        if not emit_signals:
            return

        if cutoff_fs is not None:
            self.cutoff_selected.emit(cutoff_fs)
        self.selection_changed.emit()

    def _update_cutoff_label(self, value: float | None) -> None:
        if value is None:
            self.cutoff_label.setText("Selected cutoff: None")
            return
        self.cutoff_label.setText(f"Selected cutoff: {value:.3f} fs")

    def _refresh_energy_plot(
        self,
        *,
        selected_fs: float | None = None,
    ) -> None:
        if self.energy_data is None:
            return
        self.canvas.plot_energy(
            energy_data=self.energy_data,
            suggested_fs=self._suggested_cutoff_fs,
            temp_target_k=self.temp_target_spin.value(),
            selected_fs=(
                self.get_selected_cutoff()
                if selected_fs is None
                else selected_fs
            ),
        )
