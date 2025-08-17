# -*- coding: utf-8 -*-
import sys, os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame, QToolButton, QPushButton, QSizePolicy, QGridLayout, QSlider, QRadioButton, QButtonGroup
)
from PyQt5.QtGui import QPixmap, QIcon, QMovie
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from ColorSimilarity.main_helper import *
from SSIM.ssim import get_ssim
from ObjectSimilarity.similar_image import get_best_images_annoy
from Initialization.setup import L_BINS, A_BINS, B_BINS



cwd = os.getcwd()
cost_mat_path = os.path.join(cwd, "DataBase", "emd_cost_full.npy")
ann_index_path = os.path.join(cwd, "DataBase","color_ann_index.ann")
l2_index = ann.AnnoyIndex(L_BINS*A_BINS*B_BINS, 'angular')
l2_index.load(ann_index_path)
cost_matrix = np.load(cost_mat_path)



class DragDropLabel(QLabel):
    imagePathChanged = pyqtSignal(object)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored) 
        self.current_path = None
        self._orig = None

    def dragEnterEvent(self, e):
        md = e.mimeData()
        if md.hasUrls():
            for u in md.urls():
                if u.isLocalFile() and os.path.splitext(u.toLocalFile())[1].lower() in self.exts:
                    e.acceptProposedAction(); return
        e.ignore()

    def dropEvent(self, e):
        md = e.mimeData()
        if md.hasUrls():
            for u in md.urls():
                if u.isLocalFile():
                    path = u.toLocalFile()
                    if os.path.splitext(path)[1].lower() in self.exts and os.path.isfile(path):
                        pm = QPixmap(path)
                        if not pm.isNull():
                            self._orig = pm
                            self.current_path = path
                            self.setText("")
                            self._rescale()               
                            self.imagePathChanged.emit(path)
                            e.acceptProposedAction(); return
        e.ignore()

    def resizeEvent(self, ev):
        if self._orig is not None:
            self._rescale()                          
        super().resizeEvent(ev)

    def clear_image(self, placeholder=""):
        self._orig = None
        self.current_path = None
        self.setPixmap(QPixmap())
        self.setText(placeholder)
        self.imagePathChanged.emit(None)

    def _rescale(self):
        if self._orig is None:
            return
        avail = self.contentsRect().size()          
        if avail.isEmpty():
            return
        pm = self._orig.scaled(avail, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        QLabel.setPixmap(self, pm)

class ThumbLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._orig = None
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background: rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:6px;")

    def setPixmap(self, pm: QPixmap):
        self._orig = pm if (pm and not pm.isNull()) else None
        super().setPixmap(self._scaled())

    def resizeEvent(self, e):
        if self._orig is not None:
            super().setPixmap(self._scaled())
        super().resizeEvent(e)

    def _scaled(self):
        size = self.contentsRect().size()
        if self._orig is None or size.isEmpty():
            return QPixmap()
        return self._orig.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

class FinderWorker(QThread):
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, current_paths, mode, s1_val, s2_val, parent=None):
        super().__init__(parent)
        self.current_paths = current_paths
        self.mode = mode
        self.s1_val = s1_val
        self.s2_val = s2_val

    def run(self):
        try:
            current = self.current_paths
            mode = self.mode

            if mode == "color":
                if len(current) == 2:
                    weight_img_1 = self.s1_val / 100.0
                    weight_img_2 = self.s2_val / 100.0
                    sorted_paths = color_match_double_ann(
                        img_paths=current,
                        annoy_index=l2_index,
                        l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS,
                        emd_cost_mat=cost_matrix,
                        img_weights=[weight_img_1, weight_img_2],
                        num_results=12, emd_count=12,
                        track_time=True, show=False, adjusted_bin_size=True
                    )
                else:
                    sorted_paths = color_match_single_ann(
                        img_path=current,
                        annoy_index=l2_index,
                        l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS,
                        emd_cost_mat=cost_matrix,
                        num_results=12, emd_count=12,
                        track_time=True, show=False, adjusted_bin_size=True
                    )
            
            elif mode == "ssim":
                cwd = os.getcwd()
                db_path = os.path.join(cwd, "Database", "hash_database.db")
                sorted_paths = (get_ssim(current, db_path))

            elif mode == "objects":
                cwd = os.getcwd()
                ann_file = os.path.join(cwd, "Database", "clip_embeddings.ann")
                json_file = os.path.join(cwd, "DataBase", "clip_embeddings_paths.json")
                sorted_paths = (get_best_images_annoy(current, json_file, ann_file, num_results=12))

            else:
                sorted_paths = []
                

            self.finished.emit(sorted_paths)
        except Exception as e:
            self.failed.emit(str(e))

# ------------------------------- GUI -------------------------------
class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_GUI()

    def init_GUI(self):
        # Window
        self.setWindowTitle("PicMe - Pixel-Informed Content-Matching Engine")
        self.resize(1200, 800)
        self.setWindowIcon(QIcon(os.path.join("logos", "PicMe_logo.png")))
        self.setStyleSheet("""
        QMainWindow { background-color: #011126; }
        QFrame#DashedBox {
            border: 2px dashed #9aa0a6;
            border-radius: 8px;
            background: rgba(255,255,255,0.03);
        }
        #DashedBox QLabel { color: #E6EDF3; font-size: 12pt; qproperty-alignment: AlignCenter; }
        QToolButton#ToggleBtn {
            color: #FFFFFF; background: #0E223F; border: 1px solid #3c5270; border-radius: 4px; font-weight: 700;
        }
        QToolButton#ToggleBtn:hover { background: #153055; }
        QToolButton#ToggleBtn:pressed { background: #0a1c36; }
        QPushButton#FindBtn {
            color:#fff; background:#2ea043; border:none; border-radius:8px; padding:10px 18px; font:600 13pt "Segoe UI";
        }
        QPushButton#FindBtn:hover  { background:#2aa037; }
        QPushButton#FindBtn:pressed{ background:#238636; }
        """)

        # Main layout
        central = QWidget(); self.setCentralWidget(central)
        main = QHBoxLayout(central); main.setContentsMargins(16,16,16,16); main.setSpacing(16)

        # Left column
        left_col = QWidget(); self.left_layout = QVBoxLayout(left_col)
        self.left_layout.setContentsMargins(0,0,0,0); self.left_layout.setSpacing(12)
        left_col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Right column
        right_col = QWidget(); right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0,0,0,0); right_layout.setSpacing(12)
        right_col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        main.addWidget(left_col, 3)
        main.addWidget(right_col, 7)

        # Logo
        logo_wrap = QWidget(); logo_h = QHBoxLayout(logo_wrap); logo_h.setContentsMargins(0,0,0,0)
        logo = QLabel()
        pm = QPixmap(os.path.join("logos", "PicMe_logo_cleaned.png"))
        logo.setPixmap(pm.scaled(220, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation) if not pm.isNull() else QPixmap())
        logo_h.addWidget(logo)
        self.left_layout.addWidget(logo_wrap)

        # Box 1
        self.box1_frame = QFrame(); self.box1_frame.setObjectName("DashedBox")
        box1_lay = QVBoxLayout(self.box1_frame); box1_lay.setContentsMargins(12,12,12,12)
        self.box1 = DragDropLabel("Drop image here"); box1_lay.addWidget(self.box1)
        self.box1.setMinimumHeight(160)
        self.left_layout.addWidget(self.box1_frame)
        self.box1.imagePathChanged.connect(self._maybe_show_sliders)

        # Toggle (+/-)
        self.toggle_btn = QToolButton(); self.toggle_btn.setObjectName("ToggleBtn")
        self.toggle_btn.setText("+"); self.toggle_btn.setFixedSize(28,28)
        self.toggle_btn.clicked.connect(self.toggle_second_box)
        self.left_layout.addWidget(self.toggle_btn)

        # --- Slider-Panel ---
        self.slider_panel = QWidget()
        self.slider_panel.setVisible(False)
        sp_lay = QVBoxLayout(self.slider_panel)
        sp_lay.setContentsMargins(0, 8, 0, 0)
        sp_lay.setSpacing(8)

        row1 = QHBoxLayout(); row1.setContentsMargins(0,0,0,0)
        self.s1_label = QLabel("Weight A:")
        self.s1_value = QLabel("1.00")
        self.slider1 = QSlider(Qt.Horizontal); self.slider1.setRange(0, 200); self.slider1.setSingleStep(1)
        self.slider1.valueChanged.connect(self._on_slider1_changed)
        row1.addWidget(self.s1_label); row1.addWidget(self.slider1, 1); row1.addWidget(self.s1_value)

        row2 = QHBoxLayout(); row2.setContentsMargins(0,0,0,0)
        self.s2_label = QLabel("Weight B:")
        self.s2_value = QLabel("1.00")
        self.slider2 = QSlider(Qt.Horizontal); self.slider2.setRange(0, 200); self.slider2.setSingleStep(1)
        self.slider2.valueChanged.connect(self._on_slider2_changed)
        row2.addWidget(self.s2_label); row2.addWidget(self.slider2, 1); row2.addWidget(self.s2_value)

        sp_lay.addLayout(row1)
        sp_lay.addLayout(row2)
        for lbl in (self.s1_label, self.s1_value, self.s2_label, self.s2_value):
            lbl.setStyleSheet("color: #FFFFFF; font-size: 11pt;")

        # --- Multiple-Choice---
        self.choice_panel = QWidget()
        choice_row = QHBoxLayout(self.choice_panel)
        choice_row.setContentsMargins(0, 4, 0, 0)

        self.rb_color   = QRadioButton("Color")
        self.rb_objects = QRadioButton("Objects")
        self.rb_ssim    = QRadioButton("SSIM")

        # Styling
        for rb in (self.rb_color, self.rb_objects, self.rb_ssim):
            rb.setStyleSheet("color:#FFFFFF; font-size:12pt;")

        self.choice_group = QButtonGroup(self)
        self.choice_group.addButton(self.rb_color,   0)
        self.choice_group.addButton(self.rb_objects, 1)
        self.choice_group.addButton(self.rb_ssim,    2)

        self.rb_color.setProperty("value", "color")
        self.rb_objects.setProperty("value", "objects")
        self.rb_ssim.setProperty("value", "ssim")
        self.rb_color.setChecked(True)

        choice_row.addWidget(self.rb_color)
        choice_row.addWidget(self.rb_objects)
        choice_row.addWidget(self.rb_ssim)

        idx_toggle = self.left_layout.indexOf(self.toggle_btn)
        self.left_layout.insertWidget(idx_toggle + 1, self.choice_panel)


        idx_toggle = self.left_layout.indexOf(self.toggle_btn)
        self.left_layout.insertWidget(idx_toggle + 1, self.slider_panel)

        self.second_box_frame = None
        self.second_box = None

        self.left_layout.addStretch()

        # Bottom Button
        bottom = QWidget(); bh = QHBoxLayout(bottom); bh.setContentsMargins(0,8,0,0)
        self.find_btn = QPushButton("Find best matches"); self.find_btn.setObjectName("FindBtn"); self.find_btn.setMinimumHeight(46)
        self.find_btn.clicked.connect(self.on_find_best_matches)
        bh.addWidget(self.find_btn)
        self.left_layout.addWidget(bottom)

        # Right placeholder + Grid
        self.right_placeholder = QLabel("Best fitting images"); self.right_placeholder.setAlignment(Qt.AlignCenter)
        self.right_placeholder.setStyleSheet("color:#E6EDF3; font-size:18px;")
        right_layout.addWidget(self.right_placeholder)

        # ----- Loader (GIF + "loading")-----
        self.right_loader = QWidget()
        _rl = QVBoxLayout(self.right_loader)
        _rl.setContentsMargins(0, 20, 0, 0)
        _rl.setSpacing(6)

        self.loader_gif = QLabel(); self.loader_gif.setAlignment(Qt.AlignCenter)
        gif_path = "loading.gif"  
        self.loader_movie = QMovie(gif_path)
        self.loader_gif.setMovie(self.loader_movie)

        self.loader_txt = QLabel("loading")
        self.loader_txt.setAlignment(Qt.AlignCenter)
        self.loader_txt.setStyleSheet("color:#E6EDF3; font-size:14pt;")

        _rl.addWidget(self.loader_gif, alignment=Qt.AlignHCenter)
        _rl.addWidget(self.loader_txt, alignment=Qt.AlignHCenter)

        self.right_loader.setVisible(False)
        right_layout.addWidget(self.right_loader, alignment=Qt.AlignCenter)
        # ---------------------------------------------------------------

        self.grid_widget = QWidget(); self.grid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_grid = QGridLayout(self.grid_widget); self.right_grid.setContentsMargins(0,0,0,0); self.right_grid.setSpacing(10)
        right_layout.addWidget(self.grid_widget, 1)
        self.grid_widget.setVisible(False)

        self.grid_labels = []    
        self.match_paths = []    

        self._reset_sliders()

    # ---------- Slider----------
    def _reset_sliders(self):
        self.slider1.blockSignals(True); self.slider2.blockSignals(True)
        self.slider1.setValue(100)
        self.slider2.setValue(100)
        self.slider1.blockSignals(False); self.slider2.blockSignals(False)
        self.s1_value.setText("1.00")
        self.s2_value.setText("1.00")

    def _on_slider1_changed(self, v):
        comp = 200 - v
        self.slider2.blockSignals(True)
        self.slider2.setValue(comp)
        self.slider2.blockSignals(False)
        self.s1_value.setText(f"{v/100:.2f}")
        self.s2_value.setText(f"{comp/100:.2f}")

    def _on_slider2_changed(self, v):
        comp = 200 - v
        self.slider1.blockSignals(True)
        self.slider1.setValue(comp)
        self.slider1.blockSignals(False)
        self.s2_value.setText(f"{v/100:.2f}")
        self.s1_value.setText(f"{comp/100:.2f}")

    def _maybe_show_sliders(self, *_):
        have1 = getattr(self.box1, "current_path", None) is not None
        have2 = (self.second_box is not None) and (getattr(self.second_box, "current_path", None) is not None)
        self.slider_panel.setVisible(have1 and have2)

    # --- Build 4x3 grid once ---
    def _build_grid(self, rows=4, cols=3):
        if self.grid_labels:
            return
        for idx in range(rows*cols):
            lbl = ThumbLabel()
            self.right_grid.addWidget(lbl, idx // cols, idx % cols)
            self.grid_labels.append(lbl)

    # --- Set images into grid from a list of paths (sorted left→right, top→bottom) ---
    def set_grid_images(self, paths):
        paths = [p for p in paths if p]
        for i, lbl in enumerate(self.grid_labels):
            if i < len(paths) and os.path.isfile(paths[i]):
                pm = QPixmap(paths[i])
                lbl.setPixmap(pm)
            else:
                lbl.setPixmap(QPixmap())

    def set_match_paths(self, paths):
        self.match_paths = list(paths or [])

    def _ensure_left_previews(self):
        def refresh(lbl: QLabel):
            if not lbl:
                return
        
            if getattr(lbl, "_orig", None) is not None:
                lbl._rescale()
                return
           
            path = getattr(lbl, "current_path", None)
            if path:
                pm = QPixmap(path)
                if not pm.isNull():
                    lbl._orig = pm
                    lbl.setText("")
                    lbl._rescale()
        refresh(getattr(self, "box1", None))
        refresh(getattr(self, "second_box", None))

    # --- Button handler ---
    def on_find_best_matches(self):
        # 1) Status lesen
        current = self.get_current_paths()
        mode = self.get_selected_mode()
        print("Mode:", mode)
        print("Files:", current if current else "Keine Dateien gedroppt.")

        if not self.grid_labels:
            self._build_grid(rows=4, cols=3)
        if self.right_placeholder.isVisible():
            self.right_placeholder.setVisible(False)

        self.grid_widget.setVisible(False)  
        self.right_loader.setVisible(True)
        self.loader_movie.start()

        self._worker = FinderWorker(
            current_paths=current,
            mode=mode,
            s1_val=self.slider1.value(),
            s2_val=self.slider2.value()
        )
        self._worker.finished.connect(self._on_find_done)
        self._worker.failed.connect(self._on_find_failed)
        self._worker.start()


    def get_current_paths(self):
        out = []
        if getattr(self.box1, "current_path", None):
            out.append(self.box1.current_path)
        if self.second_box and getattr(self.second_box, "current_path", None):
            out.append(self.second_box.current_path)
        return out

    def toggle_second_box(self):
        if self.second_box is None:
            self.second_box_frame = QFrame(); self.second_box_frame.setObjectName("DashedBox")
            box2_lay = QVBoxLayout(self.second_box_frame); box2_lay.setContentsMargins(12,12,12,12)
            self.second_box = DragDropLabel("Add second image")
            self.second_box.setMinimumHeight(160)
            box2_lay.addWidget(self.second_box)

            idx = self.left_layout.indexOf(self.box1_frame)
            self.left_layout.insertWidget(idx + 1, self.second_box_frame)

            self.second_box.imagePathChanged.connect(self._maybe_show_sliders)

            self.toggle_btn.setText("−")
        else:
          
            self.left_layout.removeWidget(self.second_box_frame)
            self.second_box_frame.deleteLater()
            self.second_box_frame = None
            self.second_box = None
            self.toggle_btn.setText("+")
            self.slider_panel.setVisible(False)
    
    def get_selected_mode(self):
        btn = self.choice_group.checkedButton()
        return btn.property("value") if btn is not None else None
    
    def _on_find_done(self, sorted_paths):
        try:
            self.set_grid_images(sorted_paths)
            self._ensure_left_previews()
            if getattr(self, "box1", None) and getattr(self.box1, "_orig", None) is not None:
                self.box1._rescale()
            if getattr(self, "second_box", None) and getattr(self.second_box, "_orig", None) is not None:
                self.second_box._rescale()
        finally:
        
            self.loader_movie.stop()
            self.right_loader.setVisible(False)
            self.grid_widget.setVisible(True)
            self._worker = None

    def _on_find_failed(self, msg):
        print("Fehler beim Finden:", msg)
        self.loader_movie.stop()
        self.right_loader.setVisible(False)
        self.grid_widget.setVisible(False)
        self._worker = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(os.path.join("logos", "PicMe_logo.png")))
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())
