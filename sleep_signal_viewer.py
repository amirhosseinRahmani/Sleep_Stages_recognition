import sys
import time
from pathlib import Path

import numpy as np
import pyedflib
import torch
from torch import nn

import tkinter as tk
from tkinter import filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = Path("best_sleep_lstm_standard.pth")
EPOCH_SECONDS = 30
MODEL_FS = 100
MODEL_CHANNELS = [
    "EEG Fpz-Cz",
    "EEG Pz-Oz",
    "EOG horizontal",
]
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

# Nice colors for the five sleep stages.
STAGE_COLORS = {
    "W": "#00C2FF",
    "N1": "#9B7EDE",
    "N2": "#3DDC97",
    "N3": "#2D7FF9",
    "REM": "#FF4D8D",
}

BACKGROUND = "#111827"
PANEL = "#182235"
TEXT = "#E5E7EB"
MUTED = "#94A3B8"
ACCENT = "#00C2FF"


# ============================================================
# MODEL
# ============================================================

class SleepLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(64, 5)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        return self.fc(output)


class SleepModel:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SleepLSTM().to(self.device)
        self._load()

    def _load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found:\n{self.model_path.resolve()}\n\n"
                "Put best_sleep_lstm_standard.pth beside this program."
            )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Some checkpoints saved by wrappers contain a "module." prefix.
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[7:]
            cleaned[key] = value

        self.model.load_state_dict(cleaned, strict=True)
        self.model.eval()

    def predict_epoch(self, epoch_data):
        # epoch_data shape: (3000, 3)
        x = np.asarray(epoch_data, dtype=np.float32)
        x = np.array(x, dtype=np.float32, copy=True)

        tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            prediction = int(output.argmax(dim=1).item())

        return prediction


# ============================================================
# EDF READER
# ============================================================

class EDFData:
    def __init__(self, path):
        self.path = Path(path)
        self.labels = []
        self.signals = []
        self.model_data = None
        self.duration = 0.0
        self.model_epochs = 0

        self._read()

    @staticmethod
    def _find_channel(reader, wanted):
        wanted = wanted.lower().strip()
        labels = reader.getSignalLabels()

        for index, label in enumerate(labels):
            if label.lower().strip() == wanted:
                return index

        return None

    def _read(self):
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        reader = pyedflib.EdfReader(str(self.path))

        try:
            signal_count = reader.signals_in_file
            labels = reader.getSignalLabels()

            # Read every signal for visualization.
            for index in range(signal_count):
                label = labels[index]
                fs = float(reader.getSampleFrequency(index))
                signal = reader.readSignal(index).astype(np.float32)

                self.signals.append({
                    "label": label,
                    "fs": fs,
                    "data": signal,
                })

            # Duration is based on the longest available signal.
            if self.signals:
                self.duration = max(
                    len(item["data"]) / item["fs"]
                    for item in self.signals
                    if item["fs"] > 0
                )

            # Prepare the exact three-channel input used during baseline training.
            model_signals = []
            missing = []

            for channel in MODEL_CHANNELS:
                index = self._find_channel(reader, channel)

                if index is None:
                    missing.append(channel)
                    continue

                fs = float(reader.getSampleFrequency(index))

                if abs(fs - MODEL_FS) > 0.01:
                    raise RuntimeError(
                        f"Model channel has wrong sampling rate:\n"
                        f"{channel}: {fs} Hz\n"
                        f"Expected: {MODEL_FS} Hz"
                    )

                signal = reader.readSignal(index).astype(np.float32)
                model_signals.append(signal)

            if missing:
                raise RuntimeError(
                    "These model channels were not found in the EDF file:\n\n"
                    + "\n".join(missing)
                )

            common = min(len(signal) for signal in model_signals)
            samples_per_epoch = MODEL_FS * EPOCH_SECONDS
            self.model_epochs = common // samples_per_epoch
            usable = self.model_epochs * samples_per_epoch

            epoch_signals = []

            for signal in model_signals:
                signal = signal[:usable]

                mean = float(signal.mean())
                std = float(signal.std())

                if std < 1e-8:
                    std = 1.0

                signal = (signal - mean) / std
                signal = signal.reshape(self.model_epochs, samples_per_epoch)
                epoch_signals.append(signal)

            self.model_data = np.stack(epoch_signals, axis=2).astype(np.float32)

        finally:
            reader.close()


# ============================================================
# MAIN WINDOW
# ============================================================

class SleepViewer(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Sleep Signal Viewer — LSTM Online Sleep Staging")
        self.geometry("1500x900")
        self.minsize(1100, 700)
        self.configure(bg=BACKGROUND)

        self.edf = None
        self.model = None
        self.predictions = {}

        self.current_time = 0.0
        self.view_seconds = 60.0
        self.playing = False
        self.last_update = None
        self.timer_id = None
        self.last_predicted_epoch = -1

        self.fig = None
        self.canvas = None
        self.axes = []
        self.lines = []
        self.epoch_patches = []
        self.epoch_lines = []

        self._build_ui()
        self._load_model()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def _build_ui(self):
        top = tk.Frame(self, bg=BACKGROUND)
        top.pack(fill="x", padx=18, pady=(15, 8))

        title = tk.Label(
            top,
            text="PSG Sleep Signal Viewer",
            font=("Segoe UI", 22, "bold"),
            bg=BACKGROUND,
            fg=TEXT,
        )
        title.pack(side="left")

        self.status_label = tk.Label(
            top,
            text="No EDF loaded",
            font=("Segoe UI", 10),
            bg=BACKGROUND,
            fg=MUTED,
        )
        self.status_label.pack(side="right", padx=10)

        controls = tk.Frame(self, bg=PANEL)
        controls.pack(fill="x", padx=18, pady=8)

        self._button(controls, "Open EDF", self.open_edf).pack(side="left", padx=8, pady=10)
        self.play_button = self._button(controls, "▶  Play", self.toggle_play)
        self.play_button.pack(side="left", padx=8, pady=10)

        self._button(controls, "⏮  Start", self.go_start).pack(side="left", padx=8, pady=10)
        self._button(controls, "⏪  -30 sec", lambda: self.jump(-30)).pack(side="left", padx=8, pady=10)
        self._button(controls, "30 sec  ⏩", lambda: self.jump(30)).pack(side="left", padx=8, pady=10)

        tk.Label(
            controls,
            text="Window:",
            font=("Segoe UI", 10, "bold"),
            bg=PANEL,
            fg=TEXT,
        ).pack(side="left", padx=(25, 5))

        self.window_var = tk.StringVar(value="60")
        window_menu = tk.OptionMenu(
            controls,
            self.window_var,
            "30",
            "60",
            "120",
            "300",
            command=self.change_window,
        )
        window_menu.config(
            bg="#24314A",
            fg=TEXT,
            activebackground="#334568",
            activeforeground=TEXT,
            highlightthickness=0,
            bd=0,
        )
        window_menu["menu"].config(bg="#24314A", fg=TEXT)
        window_menu.pack(side="left", padx=5, pady=10)

        self.time_label = tk.Label(
            controls,
            text="00:00:00 / 00:00:00",
            font=("Consolas", 11, "bold"),
            bg=PANEL,
            fg=ACCENT,
        )
        self.time_label.pack(side="right", padx=15)

        # Current predicted stage card.
        prediction_frame = tk.Frame(self, bg=BACKGROUND)
        prediction_frame.pack(fill="x", padx=18, pady=(3, 8))

        self.prediction_label = tk.Label(
            prediction_frame,
            text="Prediction: —",
            font=("Segoe UI", 17, "bold"),
            bg=PANEL,
            fg=TEXT,
            padx=20,
            pady=10,
        )
        self.prediction_label.pack(side="left", fill="x", expand=True)

        self.model_label = tk.Label(
            prediction_frame,
            text="Model: loading...",
            font=("Segoe UI", 10),
            bg=PANEL,
            fg=MUTED,
            padx=20,
            pady=10,
        )
        self.model_label.pack(side="right")

        # Plot area.
        self.plot_frame = tk.Frame(self, bg=BACKGROUND)
        self.plot_frame.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        self._show_empty_plot()

    def _button(self, parent, text, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 10, "bold"),
            bg="#24314A",
            fg=TEXT,
            activebackground="#355174",
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=12,
            pady=7,
            cursor="hand2",
        )

    def _show_empty_plot(self):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        fig = Figure(figsize=(12, 7), dpi=100, facecolor=BACKGROUND)
        ax = fig.add_subplot(111)
        ax.set_facecolor(PANEL)
        ax.text(
            0.5,
            0.5,
            "Open an EDF file to start",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=20,
            color=MUTED,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        self.fig = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    def _load_model(self):
        try:
            self.model = SleepModel(MODEL_FILE)
            self.model_label.config(
                text=f"Model: {MODEL_FILE.name} | {self.model.device}"
            )
        except Exception as exc:
            self.model_label.config(text="Model: not loaded")
            self.after(
                300,
                lambda: messagebox.showerror(
                    "Model error",
                    str(exc),
                ),
            )

    # --------------------------------------------------------
    # EDF
    # --------------------------------------------------------

    def open_edf(self):
        path = filedialog.askopenfilename(
            title="Select EDF file",
            filetypes=[
                ("EDF files", "*.edf"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        self.stop()

        try:
            self.status_label.config(text="Loading EDF...", fg=ACCENT)
            self.update_idletasks()

            self.edf = EDFData(path)
            self.predictions.clear()
            self.last_predicted_epoch = -1
            self.current_time = 0.0

            self._build_plots()
            self.status_label.config(
                text=f"Loaded: {self.edf.path.name}",
                fg="#7EE787",
            )
            self._update_view()

        except Exception as exc:
            self.edf = None
            self.status_label.config(text="EDF load failed", fg="#FF6B6B")
            messagebox.showerror("EDF error", str(exc))

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    def _build_plots(self):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        count = len(self.edf.signals)
        height = max(7, count * 2.0)

        fig = Figure(figsize=(13, height), dpi=100, facecolor=BACKGROUND)
        axes = fig.subplots(count, 1, sharex=True)

        if count == 1:
            axes = [axes]
        else:
            axes = list(axes)

        self.fig = fig
        self.axes = axes
        self.lines = []
        self.epoch_patches = []
        self.epoch_lines = []

        for ax, item in zip(axes, self.edf.signals):
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.grid(True, alpha=0.12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#334155")
            ax.spines["bottom"].set_color("#334155")

            ax.set_ylabel(
                f"{item['label']}\n({item['fs']:g} Hz)",
                color=TEXT,
                fontsize=8,
            )

            line, = ax.plot(
                [],
                [],
                linewidth=1.0,
                color=ACCENT,
                animated=False,
            )
            self.lines.append(line)

            patch = Rectangle(
                (0, 0),
                0,
                0,
                facecolor="#00C2FF",
                alpha=0.08,
                visible=False,
                zorder=0,
            )
            ax.add_patch(patch)
            self.epoch_patches.append(patch)

            epoch_line = ax.axvline(
                0,
                color="#FFFFFF",
                linewidth=1.4,
                alpha=0.9,
                visible=False,
            )
            self.epoch_lines.append(epoch_line)

        axes[-1].set_xlabel("Time (seconds)", color=TEXT)

        fig.subplots_adjust(
            left=0.12,
            right=0.99,
            top=0.98,
            bottom=0.05,
            hspace=0.18,
        )

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _visible_signal(self, item, start, end):
        fs = item["fs"]
        data = item["data"]

        first = max(0, int(start * fs))
        last = min(len(data), int(end * fs))

        if last <= first:
            return np.array([]), np.array([])

        values = data[first:last]

        # Keep plotting responsive for very high sample rates.
        max_points = 5000
        step = max(1, int(np.ceil(len(values) / max_points)))
        values = values[::step]

        times = start + np.arange(len(values), dtype=np.float64) * (step / fs)
        return times, values

    def _update_view(self):
        if self.edf is None:
            return

        start = max(0.0, self.current_time)
        end = min(self.edf.duration, start + self.view_seconds)

        if end <= start:
            end = start + self.view_seconds

        for ax, line, item in zip(self.axes, self.lines, self.edf.signals):
            times, values = self._visible_signal(item, start, end)
            line.set_data(times, values)
            ax.set_xlim(start, end)

            if len(values) > 0:
                low = float(np.nanmin(values))
                high = float(np.nanmax(values))
                if np.isfinite(low) and np.isfinite(high):
                    if abs(high - low) < 1e-12:
                        pad = 1.0
                    else:
                        pad = (high - low) * 0.08
                    ax.set_ylim(low - pad, high + pad)

        self._update_epoch_visuals()
        self._update_prediction()

        self.time_label.config(
            text=f"{self._format_time(self.current_time)} / "
                 f"{self._format_time(self.edf.duration)}"
        )

        self.canvas.draw_idle()

    def _update_epoch_visuals(self):
        if self.edf is None:
            return

        current_epoch = int(self.current_time // EPOCH_SECONDS)

        for patch, line, ax in zip(
            self.epoch_patches,
            self.epoch_lines,
            self.axes,
        ):
            start = current_epoch * EPOCH_SECONDS
            end = min(start + EPOCH_SECONDS, self.edf.duration)

            if current_epoch < self.edf.model_epochs:
                stage = self.predictions.get(current_epoch)
                if stage is not None:
                    name = STAGE_NAMES[stage]
                    color = STAGE_COLORS[name]
                    patch.set_facecolor(color)
                    patch.set_alpha(0.12)
                    patch.set_bounds(start, ax.get_ylim()[0], end - start, ax.get_ylim()[1] - ax.get_ylim()[0])
                    patch.set_visible(True)
                    line.set_xdata([start])
                    line.set_color(color)
                    line.set_visible(True)
                else:
                    patch.set_visible(False)
                    line.set_xdata([start])
                    line.set_color("#FFFFFF")
                    line.set_visible(True)
            else:
                patch.set_visible(False)
                line.set_visible(False)

    # --------------------------------------------------------
    # ONLINE PREDICTION
    # --------------------------------------------------------

    def _update_prediction(self):
        if self.edf is None or self.model is None:
            self.prediction_label.config(text="Prediction: model not available")
            return

        epoch = int(self.current_time // EPOCH_SECONDS)

        if epoch < 0 or epoch >= self.edf.model_epochs:
            self.prediction_label.config(text="Prediction: —")
            return

        # Predict once when playback reaches a new 30-second epoch.
        if epoch != self.last_predicted_epoch:
            try:
                prediction = self.model.predict_epoch(
                    self.edf.model_data[epoch]
                )
                self.predictions[epoch] = prediction
                self.last_predicted_epoch = epoch
            except Exception as exc:
                self.prediction_label.config(text="Prediction error")
                self.status_label.config(text=str(exc), fg="#FF6B6B")
                return

        stage = STAGE_NAMES[self.predictions[epoch]]
        color = STAGE_COLORS[stage]

        self.prediction_label.config(
            text=f"LIVE SLEEP STAGE  →  {stage}",
            fg=color,
        )

    # --------------------------------------------------------
    # PLAYBACK
    # --------------------------------------------------------

    def toggle_play(self):
        if self.edf is None:
            messagebox.showinfo("Open EDF", "First open an EDF file.")
            return

        if self.playing:
            self.stop()
        else:
            self.playing = True
            self.last_update = time.perf_counter()
            self.play_button.config(text="⏸  Pause")
            self._tick()

    def stop(self):
        self.playing = False
        self.last_update = None
        self.play_button.config(text="▶  Play")

        if self.timer_id is not None:
            self.after_cancel(self.timer_id)
            self.timer_id = None

    def _tick(self):
        if not self.playing or self.edf is None:
            return

        now = time.perf_counter()
        elapsed = now - self.last_update
        self.last_update = now

        self.current_time += elapsed

        if self.current_time >= self.edf.duration:
            self.current_time = self.edf.duration
            self._update_view()
            self.stop()
            return

        self._update_view()
        self.timer_id = self.after(50, self._tick)

    def go_start(self):
        if self.edf is None:
            return
        self.current_time = 0.0
        self.last_predicted_epoch = -1
        self._update_view()

    def jump(self, seconds):
        if self.edf is None:
            return
        self.current_time = min(
            max(0.0, self.current_time + seconds),
            max(0.0, self.edf.duration - 0.01),
        )
        self._update_view()

    def change_window(self, value):
        self.view_seconds = float(value)
        self._update_view()

    @staticmethod
    def _format_time(seconds):
        seconds = max(0, int(seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ============================================================
# START
# ============================================================

def main():
    app = SleepViewer()
    app.mainloop()


if __name__ == "__main__":
    main()
