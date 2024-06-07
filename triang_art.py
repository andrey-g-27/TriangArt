# Program for creating trianguated images from source ones.
# Controllable parameters: average triangle size & size variability.

import sys
from pathlib import PurePath

import numpy as np

from PIL import Image

from PyQt6.QtWidgets import \
    QApplication, QMainWindow, QWidget, QHBoxLayout, QGridLayout, QPushButton, \
    QLabel, QSlider, QCheckBox, QSizePolicy, QFileDialog, QDialog, QMessageBox, \
    QProgressDialog

from PyQt6.QtCore import Qt, QObject, QRect, pyqtSignal, pyqtSlot, QThread

import matplotlib as mpl
mpl.use("QtAgg")

from matplotlib import figure as mpl_fig

from matplotlib.backends.backend_qtagg import \
    FigureCanvasQTAgg as FigureCanvas

from triang_img import TriangArtProcessor

class TriLabelSlider(QWidget):
    def __init__(self, text_left, text_middle, text_right):
        super().__init__()
        self._layout = QGridLayout()
        self._label_left = QLabel(text_left)
        self._label_left.setAlignment(\
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._label_middle = QLabel(text_middle)
        self._label_middle.setAlignment(\
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self._label_right = QLabel(text_right)
        self._label_right.setAlignment(\
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(100)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(10)
        self._slider.setTracking(False)

        self._layout.addWidget(self._label_left, 0, 0, 1, 1)
        self._layout.addWidget(self._label_middle, 0, 1, 1, 1)
        self._layout.addWidget(self._label_right, 0, 2, 1, 1)
        self._layout.addWidget(self._slider, 1, 0, 1, 3)
        self.setLayout(self._layout)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self._slider.valueChanged.connect(self._value_changed_from_slider)

        self.set_value(0.0)

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = self._limit_quantize_float_val(value)
        self._slider.setValue(self._float_to_slider_val(self._value))
        self.value_changed.emit(self._value)

    def _limit_quantize_float_val(self, float_val):
        corr_value = min(max(float_val, 0.0), 1.0)
        return self._slider_val_to_float(self._float_to_slider_val(corr_value))

    def _slider_val_to_float(self, slider_val):
        # assuming slider minimum of 0 and value range of [0, 1]
        float_val = float(slider_val / self._slider.maximum())
        return float_val

    def _float_to_slider_val(self, float_val):
        # assuming slider minimum of 0 and value range of [0, 1]
        slider_value = int(round(float_val * self._slider.maximum()))
        return slider_value

    def _value_changed_from_slider(self, slider_val):
        self.set_value(self._slider_val_to_float(slider_val))

    value_changed = pyqtSignal(float)

class PolyArtMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._init_internal_vars()
        self._init_ui()
        self._init_dialogs()
        self._init_signals_slots()

        self._processing_thread = QThread()
        self._taproc.moveToThread(self._processing_thread)
        self._processing_thread.start()

        self._reset_image()

    def _init_internal_vars(self):
        self._img_src = None
        self._img_res = None
        self._taproc = TriangArtProcessor()

    def _init_ui(self):
        self._layout_main = QHBoxLayout()

        self._central_widget = QWidget()
        self._central_widget.setLayout(self._layout_main)
        self.setCentralWidget(self._central_widget)

        self._layout_controls = QGridLayout()

        self._btn_load = QPushButton("Load image")
        self._layout_controls.addWidget(self._btn_load, 0, 0, 1, 1)
        self._tls_triang_size = TriLabelSlider("Small", "Triangle size", "Large")
        self._tls_triang_size.set_value(0.5)
        self._layout_controls.addWidget(self._tls_triang_size, 1, 0, 1, 2)
        self._tls_triang_size_range = TriLabelSlider("Min", "Triangle size range", "Max")
        self._tls_triang_size_range.set_value(0.5)
        self._layout_controls.addWidget(self._tls_triang_size_range, 2, 0, 1, 2)
        self._btn_retry = QPushButton("Retry")
        self._layout_controls.addWidget(self._btn_retry, 3, 0, 1, 1)
        self._btn_save = QPushButton("Save result")
        self._layout_controls.addWidget(self._btn_save, 4, 0, 1, 1)
        self._cb_show_source = QCheckBox("Show original image")
        self._layout_controls.addWidget(self._cb_show_source, 3, 1, 1, 1)
        self._layout_controls.setRowStretch(self._layout_controls.rowCount(), 1)

        self._layout_controls.setColumnStretch(0, 0)
        self._layout_controls.setColumnStretch(1, 1)

        self._layout_main.addLayout(self._layout_controls)

        self._fig_disp = mpl_fig.Figure()
        self._canv_disp = FigureCanvas(self._fig_disp)
        self._canv_disp.setSizePolicy(\
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
        self._layout_main.addWidget(self._canv_disp)

        self._layout_main.setStretch(0, 0)
        self._layout_main.setStretch(1, 1)

        avail_geom = self.screen().availableGeometry()
        wnd_geom = QRect(avail_geom.x(), avail_geom.y(), \
            round(avail_geom.width() * 2.0 / 3.0), round(avail_geom.height() * 2.0 / 3.0))
        self.setGeometry(wnd_geom)
        self.setWindowTitle("PolyArt")

        self._input_controls = [self._btn_load, self._tls_triang_size, \
            self._tls_triang_size_range, self._btn_retry, self._btn_save, self._cb_show_source]

    def _init_dialogs(self):
        # lowercase extensions only
        # Qt doesn't handle some formats (at least JPEG) correctly if mime filters are used, ...
        # so had to use extension-based ones, but on case-sensitive OSs ...
        # all possible capitalizations have to be added manually
        self._io_file_name_filters = ["*.jpg *.jpe *.jpeg", "*.png", "*.bmp"]

        self._dlg_file_load = QFileDialog()
        self._dlg_file_load.setWindowTitle("Open file")
        self._dlg_file_load.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        self._dlg_file_load.setFileMode(QFileDialog.FileMode.ExistingFile)
        self._dlg_file_load.setNameFilters(self._io_file_name_filters)
        
        self._dlg_file_save = QFileDialog()
        self._dlg_file_save.setWindowTitle("Save file")
        self._dlg_file_save.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        self._dlg_file_save.setFileMode(QFileDialog.FileMode.AnyFile)
        self._dlg_file_save.setNameFilters(self._io_file_name_filters)

        self._dlg_progress = None

    def _init_signals_slots(self):
        self._btn_load.pressed.connect(self._dlg_file_load.open)
        self._btn_save.pressed.connect(self._dlg_file_save.open)
        self._btn_retry.pressed.connect(self._generate_result)

        self._cb_show_source.stateChanged.connect(self._show_image_auto)

        self._dlg_file_load.accepted.connect(self._dlg_file_load_closed)
        self._dlg_file_load.rejected.connect(self._dlg_file_load_closed)
        self._dlg_file_save.accepted.connect(self._dlg_file_save_closed)
        self._dlg_file_save.rejected.connect(self._dlg_file_save_closed)
        
        self._tls_triang_size.value_changed.connect(self._generate_result)
        self._tls_triang_size_range.value_changed.connect(self._generate_result)

        self._sig_set_source_image.connect(self._taproc.set_source_image)
        self._sig_generate_result.connect(self._taproc.generate_with_params)
        self._taproc.generated_result.connect(self._process_result)

        self._taproc.report_progress.connect(self._gather_progress)

    def _set_inputs_enabled(self, lock):
        for obj in self._input_controls:
            obj.setEnabled(lock)

    @pyqtSlot()
    def _dlg_file_load_closed(self):
        if (self._dlg_file_load.result() == QDialog.DialogCode.Rejected):
            return
        selected_files = self._dlg_file_load.selectedFiles()
        if (len(selected_files) < 1):
            return
        selected_file = selected_files[0]

        proc_error = None
        self._img_src = None
        self._img_res = None
        
        try:
            img = Image.open(selected_file)
        except Exception as ex:
            proc_error = "Can't open file."
        else:
            if (img.mode != "RGB"): # handle only uint8 RGB images (no alpha, palettes etc.)
                proc_error = "Only 24bpp RGB images are allowed."
            else:
                try:
                    img_data = img.getdata()
                except Exception as ex:
                    proc_error = "Can't read data."
                else:
                    self._img_src = np.reshape(np.array(img_data), \
                        (*img.size[-1::-1], 3)) / 255.0
            img.close()

        if (proc_error is not None):
            self._show_error_message(proc_error)
            return

        self._reset_image()
        self._show_image_auto()
        self._sig_set_source_image.emit(self._img_src)
        self._generate_result()

    @pyqtSlot()
    def _dlg_file_save_closed(self):
        if (self._img_res is None):
            return
        
        if (self._dlg_file_save.result() == QDialog.DialogCode.Rejected):
            return
        selected_files = self._dlg_file_save.selectedFiles()
        if (len(selected_files) < 1):
            return
        selected_file = selected_files[0]

        cur_file_ext = PurePath(selected_file).suffix[1::]
        selected_name_filter = self._dlg_file_save.selectedNameFilter()
        all_file_exts = selected_name_filter.replace("*.", "").split(" ")
        default_file_ext = all_file_exts[0]
        
        if (cur_file_ext == ""):
            if (default_file_ext is not None):
                selected_file = "".join([selected_file, ".", default_file_ext])
        else:
            if (not(cur_file_ext in set(all_file_exts))):
                selected_file = "".join([selected_file, ".", default_file_ext])

        img_sz = tuple(self._img_res.shape[1::-1])
        img_data = np.round(self._img_res * 255.0).astype(np.uint8).tobytes()
        try:
            img = Image.frombytes("RGB", img_sz, img_data)
            img.save(selected_file)
            img.close()
        except Exception as ex:
            self._show_error_message("Can't save file.")

    @pyqtSlot()
    def _generate_result(self):
        if (self._img_src is None):
            return
        self._set_inputs_enabled(False)
        # source image for self._taproc is always set before (or is None)
        self._sig_generate_result.emit(self._tls_triang_size.get_value(), \
            self._tls_triang_size_range.get_value())

    @pyqtSlot(np.ndarray)
    def _process_result(self, data):
        self._img_res = data
        self._show_image_auto()
        self._set_inputs_enabled(True)

    @pyqtSlot(int)
    def _gather_progress(self, val):
        if (self._dlg_progress is None): # init once
            self._dlg_progress = QProgressDialog("", "", 0, 10000)
            self._dlg_progress.setWindowModality(Qt.WindowModality.WindowModal)
            self._dlg_progress.setWindowTitle("Processing image")
            self._dlg_progress.setMinimumDuration(0)
            self._dlg_progress.setCancelButton(None)

        self._dlg_progress.setValue(val)

    @pyqtSlot(np.ndarray)
    def _show_image(self, data):
        self._reset_image()
        self._axes_disp.imshow(data)
        self._canv_disp.draw()

    @pyqtSlot()
    def _reset_image(self):
        self._fig_disp.clear()
        self._fig_disp.set_tight_layout(True)
        self._axes_disp = self._fig_disp.add_subplot(1, 1, 1)
        self._axes_disp.set_axis_off()
        self._canv_disp.draw()

    @pyqtSlot()
    def _show_image_auto(self):
        if (self._cb_show_source.isChecked()):
            if (self._img_src is not None):
                self._show_image(self._img_src)
        else:
            if (self._img_res is not None):
                self._show_image(self._img_res)

    def _show_error_message(self, text):
        mbox = QMessageBox()
        mbox.setText(text)
        mbox.setIcon(QMessageBox.Icon.Critical)
        mbox.exec()

    _sig_set_source_image = pyqtSignal(np.ndarray)
    _sig_generate_result = pyqtSignal(float, float)

app = QApplication(sys.argv)
wnd = PolyArtMainWindow()
wnd.show()
sys.exit(app.exec())
