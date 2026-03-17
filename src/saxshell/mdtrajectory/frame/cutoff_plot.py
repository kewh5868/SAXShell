from __future__ import annotations

import tkinter as tk
from collections.abc import Callable

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure

from .cp2k_ener import CP2KEnergyData
from .cutoff_analysis import SteadyStateResult


class CP2KEnergyCutoffSelector:
    """Tkinter window for interactive cutoff selection on CP2K .ener
    data.

    This is a temporary UI implementation. The data and analysis logic
    live outside this class so the plotting layer can later be replaced
    by Qt without affecting the core workflow.
    """

    def __init__(
        self,
        energy_data: CP2KEnergyData,
        steady_state: SteadyStateResult | None = None,
    ) -> None:
        self.energy_data = energy_data
        self.steady_state = steady_state
        self.cutoff_x_fs: float | None = None
        self._dragging = False
        self._cutoff_lines = []

    def show(
        self,
        on_cutoff: Callable[[float], None] | None = None,
    ) -> None:
        """Show the interactive cutoff selector window."""
        win = tk.Tk()
        win.title("CP2K Energy Cutoff Selector")

        fig = Figure(figsize=(8, 6), dpi=100)
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312, sharex=ax1)
        ax3 = fig.add_subplot(313, sharex=ax1)

        time_fs = self.energy_data.time_fs
        kinetic = self.energy_data.kinetic
        temperature = self.energy_data.temperature
        potential = self.energy_data.potential

        ax1.plot(time_fs, kinetic, color="red", label="Kinetic Energy")
        ax2.plot(time_fs, temperature, color="blue", label="Temperature")
        ax3.plot(time_fs, potential, color="green", label="Potential Energy")

        ax1.set_ylabel("Kinetic")
        ax2.set_ylabel("Temp (K)")
        ax3.set_ylabel("Potential")
        ax3.set_xlabel("Time (fs)")

        if self.steady_state is not None:
            if self.steady_state.cutoff_time_fs is not None:
                for ax in (ax1, ax2, ax3):
                    ax.axvline(
                        self.steady_state.cutoff_time_fs,
                        linestyle="--",
                        color="black",
                        label=(
                            "Suggested cutoff = "
                            f"{self.steady_state.cutoff_time_fs:.2f} fs"
                        ),
                    )

            ax2.axhline(
                self.steady_state.temp_target_k,
                linestyle="-",
                color="gray",
                label=(
                    "Target temperature = "
                    f"{self.steady_state.temp_target_k:.1f} K"
                ),
            )

        mid_fs = float(time_fs[len(time_fs) // 2]) if len(time_fs) > 0 else 0.0
        self.cutoff_x_fs = mid_fs
        self._cutoff_lines = [
            ax.axvline(
                mid_fs,
                color="magenta",
                linestyle="--",
                linewidth=2,
            )
            for ax in (ax1, ax2, ax3)
        ]

        def on_press(event) -> None:
            if event.xdata is None:
                return

            x_range = float(time_fs.max() - time_fs.min())
            tol = max(x_range * 0.01, 1.0e-12)

            for line in self._cutoff_lines:
                x0 = float(line.get_xdata()[0])
                if abs(float(event.xdata) - x0) < tol:
                    self._dragging = True
                    break

        def on_motion(event) -> None:
            if not self._dragging or event.xdata is None:
                return
            self.cutoff_x_fs = float(event.xdata)
            for line in self._cutoff_lines:
                line.set_xdata([self.cutoff_x_fs, self.cutoff_x_fs])
            fig.canvas.draw_idle()

        def on_release(event) -> None:
            self._dragging = False
            if event.xdata is not None:
                self.cutoff_x_fs = float(event.xdata)

        fig.canvas.mpl_connect("button_press_event", on_press)
        fig.canvas.mpl_connect("motion_notify_event", on_motion)
        fig.canvas.mpl_connect("button_release_event", on_release)

        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")
        ax3.legend(loc="upper right")
        fig.tight_layout(pad=2.0)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()

        btn_frame = tk.Frame(win)
        btn_frame.pack(side=tk.BOTTOM, pady=5)

        tk.Button(
            btn_frame,
            text="Close Window",
            command=win.destroy,
        ).pack(side=tk.LEFT, padx=5)

        def _internal_set_cutoff() -> None:
            if self.cutoff_x_fs is None:
                return
            if callable(on_cutoff):
                on_cutoff(self.cutoff_x_fs)
            print(f"Cutoff set at {self.cutoff_x_fs:.2f} fs")

        tk.Button(
            btn_frame,
            text="Set Cutoff",
            command=_internal_set_cutoff,
        ).pack(side=tk.LEFT)

        win.mainloop()
