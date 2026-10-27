# SPDX-License-Identifier: AGPL-3.0-or-later
"""Save folders and processing limits, with an optional bounded CPU benchmark."""
import fcntl
import shutil

from PySide6.QtCore import QProcess, QTimer
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QFileDialog, QFrame, QWidget, QTabWidget, QScrollArea,
                               QFormLayout, QDoubleSpinBox, QSpinBox)
from app2 import library_dir
from app2.builder import hardware_benchmark as benchmark
from app2.qt.theme import T
from app2.qt.widgets import ElidedLabel
from scanner.stages import resources


class PreferencesDialog(QDialog):
    ROWS = (("projects", "Projects", "Scans, and projects imported from the scanner. Takes effect the next time PointYoink starts."),
            ("exports", "Exports", "Where Export suggests saving models. You can still pick any folder when you save."),
            ("screenshots", "Screenshots", "Pictures copied off the scanner's screen."),
            ("recordings", "Screen recordings", "Screen recordings copied off the scanner."))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(620)
        self.resize(720, 720)
        self._benchmark = None
        self._stopper = None
        self._checking_service = False
        self._benchmark_lock = None
        self._benchmark_unit = None
        self._canceling = False
        self._service_wait_lost = False
        self._closing_result = None
        self._gpu = None
        self.setStyleSheet("QDialog {background:%s;} QLabel {color:%s; background:transparent;} QLabel[role=\"hint\"] {color:%s;}" % (T["window"], T["ink"], T["ink2"]))
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 18, 20, 16)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("preferences_tabs")
        layout.addWidget(self.tabs, 1)
        self._general_tab()
        self._folders_tab()
        self._processing_tab()
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close)
        layout.addLayout(row)
        self._gpu_timer = QTimer(self)
        self._gpu_timer.setSingleShot(True)
        self._gpu_timer.timeout.connect(self._gpu_timeout)
        self._benchmark_timer = QTimer(self)
        self._benchmark_timer.setSingleShot(True)
        self._benchmark_timer.timeout.connect(self._benchmark_timeout)
        self._stop_timer = QTimer(self)
        self._stop_timer.setSingleShot(True)
        self._stop_timer.timeout.connect(self._stop_timeout)
        self._memory_timer = QTimer(self)
        self._memory_timer.setInterval(3000)
        self._memory_timer.timeout.connect(self._update_hardware)
        self._memory_timer.start()
        self.refresh()
        self._update_hardware()
        previous = benchmark.load_result()
        if previous:
            self._show_result(previous, "Last recorded test")
        QTimer.singleShot(0, self._start_gpu_query)

    def _tab(self, name):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(10)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, name)
        return layout

    @staticmethod
    def _hint(text):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setProperty("role", "hint")
        return label

    def _general_tab(self):
        """What the app does by itself: the look a project opens with, the fill light reminder, the timelapse mark, and
        where the app starts."""
        from PySide6.QtWidgets import QCheckBox, QComboBox
        from app2 import general_prefs as G
        g = G.load(); layout = self._tab("General")
        def box(title, hint):
            frame = QFrame(); frame.setObjectName("PrefBox"); frame.setStyleSheet("QFrame#PrefBox {border:1px solid #3a414b; border-radius:8px;}")
            col = QVBoxLayout(frame); col.setContentsMargins(14, 10, 14, 12); col.setSpacing(6)
            name = QLabel(title); name.setStyleSheet("font-weight:600;"); col.addWidget(name); col.addWidget(self._hint(hint))
            layout.addWidget(frame); return col
        col = box("Model look when a project opens", "Scan color is one click away on the Look button either way.")
        self.default_look = QComboBox(); self.default_look.addItem("Shaded", "shaded"); self.default_look.addItem("Scan color", "scan_color")
        self.default_look.setCurrentIndex(max(0, self.default_look.findData(g["default_look"]))); col.addWidget(self.default_look)
        self.default_look.currentIndexChanged.connect(lambda i: G.save(default_look=self.default_look.itemData(i)))
        col = box("Fill light reminder", "The scanner's fill light stays on while you scan in color. After this long idle, the app asks whether you still need it, then turns it off.")
        row = QHBoxLayout(); self.light_minutes = QSpinBox(); self.light_minutes.setRange(1, 60); self.light_minutes.setSuffix(" min"); self.light_minutes.setValue(g["light_minutes"])
        self.light_minutes.valueChanged.connect(lambda v: G.save(light_minutes=int(v))); row.addWidget(self.light_minutes); row.addStretch(1); col.addLayout(row)
        col = box("Scan timelapse", "The PointYoink mark in the corner of exported GIFs and videos. The export sheet can still change it each time.")
        self.timelapse_watermark = QCheckBox("Watermark on by default"); self.timelapse_watermark.setChecked(g["timelapse_watermark"]); col.addWidget(self.timelapse_watermark)
        self.timelapse_watermark.toggled.connect(lambda on: G.save(timelapse_watermark=bool(on)))
        col = box("When PointYoink starts", "Takes effect the next time PointYoink starts.")
        self.open_last = QCheckBox("Open the newest project instead of the Projects page"); self.open_last.setChecked(g["open_last"]); col.addWidget(self.open_last)
        self.open_last.toggled.connect(lambda on: G.save(open_last=bool(on)))
        layout.addStretch(1)

    def _folders_tab(self):
        layout = self._tab("Save folders")
        self.labels = {}
        for key, title, hint in self.ROWS:
            box = QFrame()
            box.setObjectName("PrefBox")
            box.setStyleSheet("QFrame#PrefBox {border:1px solid #3a414b; border-radius:8px;}")
            column = QVBoxLayout(box)
            column.setContentsMargins(14, 10, 14, 12)
            column.setSpacing(6)
            name = QLabel(title)
            name.setStyleSheet("font-weight:600;")
            column.addWidget(name)
            column.addWidget(self._hint(hint))
            row = QHBoxLayout()
            label = ElidedLabel()
            self.labels[key] = label
            row.addWidget(label, 1)
            change = QPushButton("Change...")
            change.setObjectName("change_" + key + "_folder")
            change.clicked.connect(lambda _checked=False, k=key: self.change(k))
            row.addWidget(change)
            column.addLayout(row)
            layout.addWidget(box)
        layout.addStretch(1)

    def _processing_tab(self):
        layout = self._tab("Processing")
        title = QLabel("Processing resources")
        title.setStyleSheet("font-size:18px; font-weight:600;")
        layout.addWidget(title)
        layout.addWidget(self._hint("Limits apply to staged builds, edits, and previews. Live scanning and the app window use additional resources. Running jobs keep their current limits."))
        self.memory_info = QLabel()
        self.memory_info.setWordWrap(True)
        self.memory_info.setObjectName("processing_memory_info")
        layout.addWidget(self.memory_info)
        self.gpu_info = self._hint("Checking NVIDIA GPU information...")
        self.gpu_info.setObjectName("processing_gpu_info")
        layout.addWidget(self.gpu_info)
        layout.addWidget(self._hint("GPU RAM is separate from system RAM. These controls do not cap GPU utilization."))
        form = QFormLayout()
        settings = resources.load()
        maximum = resources.memory_max_gb()
        self.memory_limit = QDoubleSpinBox()
        self.memory_limit.setObjectName("processing_ram_budget")
        self.memory_limit.setDecimals(1)
        self.memory_limit.setRange(1, max(1, maximum))
        self.memory_limit.setSingleStep(.5)
        self.memory_limit.setSuffix(" GiB")
        self.memory_limit.setValue(settings["memory_gb"])
        self.memory_limit.setEnabled(maximum >= 1)
        form.addRow("Processing RAM limit", self.memory_limit)
        self.cpu_limit = QSpinBox()
        self.cpu_limit.setObjectName("processing_cpu_budget")
        self.cpu_limit.setRange(10, 80)
        self.cpu_limit.setSuffix(" %")
        self.cpu_limit.setValue(settings["cpu_percent"])
        form.addRow("CPU share", self.cpu_limit)
        layout.addLayout(form)
        self.reserve_info = self._hint("")
        layout.addWidget(self.reserve_info)
        save_row = QHBoxLayout()
        save_row.addStretch(1)
        self.save_button = QPushButton("Save limits")
        self.save_button.setObjectName("processing_settings_save")
        self.save_button.setEnabled(maximum >= 1)
        self.save_button.clicked.connect(self._save_limits)
        save_row.addWidget(self.save_button)
        layout.addLayout(save_row)
        self.settings_status = self._hint("")
        self.settings_status.setObjectName("processing_settings_status")
        layout.addWidget(self.settings_status)
        self.settings_status.hide()
        self.recommendation_info = self._hint("")
        self.recommendation_info.setObjectName("processing_recommendation")
        layout.addWidget(self.recommendation_info)
        self.apply_button = QPushButton("Apply recommendation")
        self.apply_button.setObjectName("apply_hardware_recommendation")
        self.apply_button.clicked.connect(self._apply_recommendation)
        layout.addWidget(self.apply_button)
        layout.addWidget(self._hint("Optional synthetic CPU test: 1 GiB maximum, at most two CPU cores, and 30 seconds. No scans or GPU allocations. Timing does not predict full-scan speed or change model quality."))
        controls = QHBoxLayout()
        self.benchmark_button = QPushButton("Run CPU benchmark")
        self.benchmark_button.setObjectName("run_hardware_benchmark")
        self.benchmark_button.clicked.connect(self._start_benchmark)
        controls.addWidget(self.benchmark_button)
        self.cancel_button = QPushButton("Cancel benchmark")
        self.cancel_button.setObjectName("cancel_hardware_benchmark")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_benchmark)
        controls.addWidget(self.cancel_button)
        layout.addLayout(controls)
        self.benchmark_status = self._hint("No benchmark has been run. Recommendations use available RAM and CPU count.")
        self.benchmark_status.setObjectName("hardware_benchmark_status")
        layout.addWidget(self.benchmark_status)
        layout.addStretch(1)

    def _suggestion(self):
        info = resources.memory_info()
        return benchmark.recommendation(*info, resources.cpu_count()) if info else None

    def _update_hardware(self):
        info = resources.memory_info()
        if info:
            total, available = (value / resources.GIB for value in info)
            reserve = resources.desktop_reserve_bytes(info[0]) / resources.GIB
            self.memory_info.setText("System RAM: %.1f GiB total, %.1f GiB available. CPU: %d logical processors." % (total, available, resources.cpu_count()))
            self.reserve_info.setText("Checks for %.1f GiB of desktop headroom before each job. RAM limits shrink when less is available. CPU share is a ceiling across logical processors; some stages use fewer cores." % reserve)
        else:
            self.memory_info.setText("System RAM information is unavailable.")
            self.reserve_info.setText("Processing cannot start safely until available RAM can be read.")
        suggestion = self._suggestion()
        self.apply_button.setEnabled(suggestion is not None)
        if suggestion:
            self.recommendation_info.setText("Suggested starting limits: %g GiB and %d%% CPU, based on available RAM and CPU count. Apply to replace your saved limits." % (suggestion["memory_gb"], suggestion["cpu_percent"]))
        else:
            self.recommendation_info.setText("A recommendation is unavailable while the desktop RAM reserve cannot be kept.")

    def _save_limits(self):
        self.settings_status.show()
        try:
            resources.save(self.memory_limit.value(), self.cpu_limit.value())
        except (OSError, ValueError) as error:
            self.settings_status.setText("Limits were not saved: %s" % error)
            return
        self.settings_status.setText("Saved. New processing jobs can use up to %g GiB and at most %d%% CPU." % (self.memory_limit.value(), self.cpu_limit.value()))

    def _apply_recommendation(self):
        suggestion = self._suggestion()
        if suggestion is None:
            self.settings_status.show()
            self.settings_status.setText("Available RAM changed. A safe recommendation is not available now.")
            self._update_hardware()
            return
        self.memory_limit.setValue(suggestion["memory_gb"])
        self.cpu_limit.setValue(suggestion["cpu_percent"])
        self._save_limits()

    def _start_gpu_query(self):
        if not self.isVisible():
            return
        executable = shutil.which("nvidia-smi")
        if not executable:
            self.gpu_info.setText("NVIDIA GPU information is unavailable. CPU processing is still available.")
            return
        process = QProcess(self)
        self._gpu = process
        process.finished.connect(self._gpu_finished)
        process.errorOccurred.connect(self._gpu_error)
        process.start(executable, ["--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"])
        self._gpu_timer.start(3000)

    def _gpu_error(self, _error):
        self.gpu_info.setText("GPU information could not be read. CPU processing is still available.")

    def _gpu_timeout(self):
        if self._gpu and self._gpu.state() != QProcess.ProcessState.NotRunning:
            self.gpu_info.setText("GPU information timed out. CPU processing is still available.")
            self._gpu.kill()

    def _gpu_finished(self, code, _status):
        self._gpu_timer.stop()
        process = self._gpu
        if not process:
            return
        if code == 0:
            try:
                rows = benchmark.gpu_rows(bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace"))
                self.gpu_info.setText("\n".join("%s: GPU RAM %.1f GiB total, %.1f GiB free." % (row["name"], row["total_gb"], row["free_gb"]) for row in rows))
            except ValueError as error:
                self.gpu_info.setText(str(error))
        else:
            self.gpu_info.setText("GPU information could not be read. CPU processing is still available.")
        self._gpu = None
        process.deleteLater()

    def _start_benchmark(self):
        if self._benchmark is not None:
            return
        lock = None
        try:
            args = benchmark.command(benchmark.new_unit())
            unit = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--unit="))
            resources.config_dir().mkdir(parents=True, exist_ok=True)
            lock = (resources.config_dir() / "processing.lock").open("a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("A processing job or benchmark is already running. Wait for it to finish.") from None
        except (OSError, ValueError) as error:
            if lock:
                lock.close()
            self.benchmark_status.setText("Benchmark was not started: %s" % error)
            return
        self._benchmark_lock = lock
        self._benchmark_unit = unit
        self._canceling = False
        self._service_wait_lost = False
        self.benchmark_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.benchmark_status.setText("Running the bounded CPU test. You can cancel or close this dialog to stop it.")
        process = QProcess(self)
        self._benchmark = process
        process.finished.connect(self._benchmark_finished)
        process.errorOccurred.connect(self._benchmark_error)
        process.start(args[0], args[1:])
        self._benchmark_timer.start((benchmark.MAX_SECONDS + 10) * 1000)

    def _benchmark_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.benchmark_status.setText("The capped benchmark service could not start. No uncapped test was attempted.")
            self._finish_benchmark()

    def _show_result(self, result, prefix="CPU test finished"):
        self.benchmark_status.setText("%s: 150,000 nearest-neighbor queries in %.3f s (%s queries/s), including tree builds. Single-threaded synthetic CPU throughput; recommendations still use RAM and CPU count." % (prefix, result["elapsed_seconds"], format(round(result["queries_per_second"]), ",")))

    def _benchmark_finished(self, code, status):
        process = self._benchmark
        if not process:
            return
        if status == QProcess.ExitStatus.CrashExit:
            self._service_wait_lost = True
            self._cancel_benchmark()
            return
        if self._canceling:
            self.benchmark_status.setText("Benchmark stopped. Saved processing limits were not changed.")
        elif code != 0:
            error = bytes(process.readAllStandardError()).decode("utf-8", errors="replace").strip()[-1200:]
            self.benchmark_status.setText("Benchmark could not finish safely. " + (error or "The capped service stopped without a result."))
        else:
            try:
                result = benchmark.decode_result(bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace"))
                benchmark.save_result(result)
                self._show_result(result)
            except (OSError, ValueError) as error:
                self.benchmark_status.setText("Benchmark result was not saved: %s" % error)
        self._finish_benchmark()

    def _finish_benchmark(self):
        self._benchmark_timer.stop()
        self._stop_timer.stop()
        self._checking_service = False
        if self._stopper:
            self._stopper.finished.disconnect(self._stop_finished)
            self._stopper.errorOccurred.disconnect(self._stop_error)
            if self._stopper.state() != QProcess.ProcessState.NotRunning:
                self._stopper.kill()
                self._stopper.waitForFinished(1000)
            self._stopper.deleteLater()
            self._stopper = None
        process, self._benchmark = self._benchmark, None
        if process:
            process.deleteLater()
        if self._benchmark_lock:
            self._benchmark_lock.close()
            self._benchmark_lock = None
        self._benchmark_unit = None
        self.benchmark_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._update_hardware()
        if self._closing_result is not None:
            result, self._closing_result = self._closing_result, None
            self.done(result)

    def _benchmark_timeout(self):
        self.benchmark_status.setText("The CPU test timed out. Stopping its service...")
        self._cancel_benchmark()

    def _cancel_benchmark(self):
        if self._benchmark is None or self._stopper is not None:
            return
        self._canceling = True
        self.cancel_button.setEnabled(False)
        self.benchmark_status.setText("Stopping the benchmark service...")
        self._run_service_control(benchmark.stop_command(self._benchmark_unit))

    def _run_service_control(self, args, checking=False):
        self._checking_service = checking
        process = QProcess(self)
        self._stopper = process
        process.finished.connect(self._stop_finished)
        process.errorOccurred.connect(self._stop_error)
        process.start(args[0], args[1:])
        self._stop_timer.start(5000)

    def _stop_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._stop_finished(1, QProcess.ExitStatus.NormalExit)

    def _stop_timeout(self):
        if self._stopper and self._stopper.state() != QProcess.ProcessState.NotRunning:
            self._stopper.kill()

    def _stop_finished(self, code, status):
        self._stop_timer.stop()
        process, self._stopper = self._stopper, None
        checking, self._checking_service = self._checking_service, False
        output = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace") if checking and process else ""
        if process:
            process.deleteLater()
        if self._service_wait_lost:
            confirmed = code == 0 and status == QProcess.ExitStatus.NormalExit
            if checking:
                confirmed = confirmed and benchmark.service_stopped(self._benchmark_unit, output)
            if confirmed:
                self.benchmark_status.setText("Benchmark stopped after its service connection closed.")
                self._finish_benchmark()
                return
            if not checking:
                # --collect may already have unloaded the service. A failed stop
                # is not proof either way; inspect it before releasing the lock.
                self.benchmark_status.setText("Checking whether the benchmark service has stopped...")
                self._run_service_control(benchmark.status_command(self._benchmark_unit), checking=True)
                return
        if self._benchmark is not None:
            if checking:
                self.benchmark_status.setText("Waiting for confirmation that the benchmark service stopped. Processing remains paused.")
            else:
                self.benchmark_status.setText("Waiting for the benchmark service to stop. Its runtime is limited to 30 seconds.")
            # Retry if cancellation reached systemd before service creation completed.
            self._benchmark_timer.start(2000)

    def done(self, result):
        if self._benchmark is not None:
            self._closing_result = result
            self._cancel_benchmark()
            return
        self._memory_timer.stop()
        self._gpu_timer.stop()
        if self._gpu and self._gpu.state() != QProcess.ProcessState.NotRunning:
            self._gpu.kill()
            self._gpu.waitForFinished(1000)
        super().done(result)

    def path(self, key):
        return library_dir.folder(key)

    def refresh(self):
        for key, label in self.labels.items():
            label.setText(self.path(key))
            label.setToolTip(self.path(key))

    def change(self, key):
        directory = QFileDialog.getExistingDirectory(self, "Choose a folder", self.path(key))
        if not directory:
            return
        library_dir.set_folder(key, directory)
        self.refresh()
