# -*- coding: utf-8 -*-
import sys, os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame, QToolButton, QPushButton, QSizePolicy, QGridLayout
)
from PyQt5.QtGui import QPixmap, QIcon, QImage, QFont
from PyQt5.QtCore import Qt, QSize


# -------- Drop-Label: nimmt lokale Bilddateien an und merkt den Pfad --------
class DragDropLabel(QLabel):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.current_path = None         # <-- hier merken wir den Pfad
        self._orig = None                # Original-Pixmap für sauberes Rescale

    def dragEnterEvent(self, e):
        md = e.mimeData()
        if md.hasUrls():
            for u in md.urls():
                if u.isLocalFile() and os.path.splitext(u.toLocalFile())[1].lower() in self.exts:
                    e.acceptProposedAction()
                    return
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
                            self.setText("")  # Platzhalter ausblenden
                            self._rescale()
                            e.acceptProposedAction()
                            return
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

    def _rescale(self):
        if self._orig is None or self.width() <= 0 or self.height() <= 0:
            return
        pm = self._orig.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        super().setPixmap(pm)


# -------- Thumbnail-Label fürs rechte Grid (skaliert mit) --------
class ThumbLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._orig = None
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:6px;")

    def setPixmap(self, pm: QPixmap):
        self._orig = pm if (pm and not pm.isNull()) else None
        super().setPixmap(self._scaled())

    def resizeEvent(self, e):
        if self._orig is not None:
            super().setPixmap(self._scaled())
        super().resizeEvent(e)

    def _scaled(self):
        if self._orig is None or self.width() <= 0 or self.height() <= 0:
            return QPixmap()
        return self._orig.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)


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

        # Logo (optional)
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
        self.left_layout.addWidget(self.box1_frame)

        # Toggle (+/-) für zweite Box
        self.toggle_btn = QToolButton(); self.toggle_btn.setObjectName("ToggleBtn")
        self.toggle_btn.setText("+"); self.toggle_btn.setFixedSize(28,28)
        self.toggle_btn.clicked.connect(self.toggle_second_box)
        self.left_layout.addWidget(self.toggle_btn)

        # Zweite Box (start: None)
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

        self.grid_widget = QWidget(); self.grid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_grid = QGridLayout(self.grid_widget); self.right_grid.setContentsMargins(0,0,0,0); self.right_grid.setSpacing(10)
        right_layout.addWidget(self.grid_widget, 1)
        self.grid_widget.setVisible(False)

        self.grid_labels = []    # wird beim ersten Klick gebaut
        # Liste der Match-Pfade (nur diese landen im Grid)
        self.match_paths = []

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

    # --- Extern deine Ergebnisliste setzen (optional nutzbar) ---
    def set_match_paths(self, paths):
        self.match_paths = list(paths or [])

    # --- Button handler ---
    def on_find_best_matches(self):
        # 1) Pfade der aktuell gedroppten Bilder links drucken
        current = self.get_current_paths()
        print("Files:", current if current else "Keine Dateien gedroppt.")

        # 2) Grid einmalig bauen und anzeigen
        if not self.grid_labels:
            self._build_grid(rows=4, cols=3)
        if self.right_placeholder.isVisible():
            self.right_placeholder.setVisible(False)
        self.grid_widget.setVisible(True)

        # 3) Nur DEINE Liste ins Grid laden (alphabetisch)
        paths_for_grid = sorted(self.match_paths)
        self.set_grid_images(paths_for_grid)

    # --- Aktuell gedroppte Pfade aus den linken Boxen ---
    def get_current_paths(self):
        out = []
        if getattr(self.box1, "current_path", None):
            out.append(self.box1.current_path)
        if self.second_box and getattr(self.second_box, "current_path", None):
            out.append(self.second_box.current_path)
        return out

    # --- Plus/Minus für zweite Box ---
    def toggle_second_box(self):
        if self.second_box is None:
            # Frame + Label bauen
            self.second_box_frame = QFrame(); self.second_box_frame.setObjectName("DashedBox")
            box2_lay = QVBoxLayout(self.second_box_frame); box2_lay.setContentsMargins(12,12,12,12)
            self.second_box = DragDropLabel("Add second image")
            box2_lay.addWidget(self.second_box)

            # unter Box1 einfügen
            idx = self.left_layout.indexOf(self.box1_frame)
            self.left_layout.insertWidget(idx + 1, self.second_box_frame)

            self.toggle_btn.setText("−")
        else:
            # entfernen
            self.left_layout.removeWidget(self.second_box_frame)
            self.second_box_frame.deleteLater()
            self.second_box_frame = None
            self.second_box = None
            self.toggle_btn.setText("+")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(os.path.join("logos", "PicMe_logo.png")))
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())
