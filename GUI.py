from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys, os


# Drag&Drop-Box: nur proportional skalieren (KeepAspectRatio), kein Drehen, kein Stretching.
# Vor dem ersten Drop bleibt die Box klein; erst NACH dem Drop darf sie sich leicht nach unten vergrößern.
class DragDropLabel(QLabel):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

    def __init__(self, text="Drop image here", parent=None):
        super().__init__(text, parent)
        self.setObjectName("DashedBox")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setScaledContents(False)              # Wir skalieren selbst, immer mit KeepAspectRatio
        self.setMinimumHeight(120)                 # Start klein
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._orig_pm = None                       # erst nach Drop vorhanden
        self._inset = 0                            # Bild darf die Box an 2 gegenüberliegenden Kanten berühren

    # --- Drag & Drop ---
    def dragEnterEvent(self, e):
        if self._ok(e.mimeData()):
            e.acceptProposedAction()
            self._set_drag_active(True)

    def dragMoveEvent(self, e):
        if self._ok(e.mimeData()):
            e.acceptProposedAction()

    def dragLeaveEvent(self, e):
        self._set_drag_active(False)

    def dropEvent(self, e):
        self._set_drag_active(False)
        path = self._first_local_image_path(e.mimeData())
        if path:
            # Bild laden ohne Auto-Rotation (kein Drehen)
            reader = QImageReader(path)
            reader.setAutoTransform(True)
            img = reader.read()
            if not img.isNull():
                self._orig_pm = QPixmap.fromImage(img)
                # Höhe jetzt passend machen (leicht wachsen erlaubt, aber gedeckelt)
                self._fit_height_to_image()
                self._update_scaled_pixmap()
                self.setText("")  # Platzhaltertext weg
                e.acceptProposedAction()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # Vor dem ersten Drop NICHT vergrößern – nur wenn Bild vorhanden ist.
        if self._orig_pm is not None:
            self._fit_height_to_image()
            self._update_scaled_pixmap()

    # --- Rendering: immer KeepAspectRatio (inside fit) ---
    def _update_scaled_pixmap(self):
        if self._orig_pm is None:
            return
        cr = self.contentsRect()
        target_w = max(1, cr.width()  - 2 * self._inset)
        target_h = max(1, cr.height() - 2 * self._inset)
        pm_scaled = self._orig_pm.scaled(QSize(target_w, target_h),
                                         Qt.KeepAspectRatio, Qt.SmoothTransformation)
        super().setPixmap(pm_scaled)

    # --- Boxhöhe leicht an Bild anpassen (aber nicht riesig); VOR Drop bleibt Höhe klein ---
    def _fit_height_to_image(self):
        if self._orig_pm is None or self.width() <= 0:
            return
        img_w = self._orig_pm.width()
        img_h = self._orig_pm.height()
        if img_w <= 0:
            return

        # Höhe, die nötig wäre, damit das Bild bei voller Boxbreite komplett sichtbar ist
        avail_w = max(1, self.contentsRect().width() - 2 * self._inset)
        needed_h = int((img_h / img_w) * avail_w) + 2 * self._inset

        # „Leicht“ wachsen, aber gedeckelt (z. B. max 50% der Fensterhöhe)
        win = self.window()
        max_h_cap = 260
        if isinstance(win, QWidget) and win.height() > 0:
            max_h_cap = max(120, int(win.height() * 0.18))  # vorher ~0.35
        min_h = 120

        new_h = max(min_h, min(needed_h, max_h_cap))
        self.setMinimumHeight(min_h)
        self.setMaximumHeight(max_h_cap)
        self.setFixedHeight(new_h)

    # --- Helfer ---
    def _ok(self, md: QMimeData) -> bool:
        if md.hasUrls():
            return self._first_local_image_path(md) is not None
        return md.hasImage()

    def _first_local_image_path(self, md: QMimeData):
        if not md.hasUrls():
            return None
        for url in md.urls():
            if url.isLocalFile():
                p = url.toLocalFile()
                _, e = os.path.splitext(p)
                if e.lower() in self.exts and os.path.isfile(p):
                    return p
        return None

    def _set_drag_active(self, active: bool):
        if self.property("dragActive") != active:
            self.setProperty("dragActive", active)
            self.style().unpolish(self)
            self.style().polish(self)
            self.update()


class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_GUI()

    def init_GUI(self):
        # Fenster: 75% des Bildschirms, zentriert (bleibt fix; wir ändern die Fenstergröße nie automatisch)
        screen_rect = QApplication.desktop().screenGeometry()
        window_w = int(screen_rect.width() * 0.75)
        window_h = int(screen_rect.height() * 0.75)
        x_position = (screen_rect.width() - window_w) // 2
        y_position = (screen_rect.height() - window_h) // 2

        self.setWindowTitle("PicMe - Pixel-Informed Content-Matching Engine")
        self.setGeometry(x_position, y_position, window_w, window_h)
        self.setWindowIcon(QIcon(os.path.join("logos", "PicMe_logo.png")))

        # Styles (für QLabel#DashedBox)
        self.setStyleSheet("""
        QMainWindow { background-color: #011126; }

        QLabel#DashedBox {
            border: 2px dashed #9aa0a6;
            border-radius: 8px;
            background: rgba(255,255,255,0.03);
            color: #E6EDF3;
            font-size: 12pt;
        }
        QLabel#DashedBox[dragActive="true"] {
            border-color: #4aa3ff;
            background: rgba(74,163,255,0.08);
        }

        QToolButton#ToggleBtn {
            color: #FFFFFF;
            background: #0E223F;
            border: 1px solid #3c5270;
            border-radius: 4px;
            font-weight: 700;
        }
        QToolButton#ToggleBtn:hover { background: #153055; }
        QToolButton#ToggleBtn:pressed { background: #0a1c36; }
        """)

        # Layout (links/rechts)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(16, 16, 16, 16)
        self.main_layout.setSpacing(16)

        # Linke Spalte
        self.left_col = QWidget()
        self.left_layout = QVBoxLayout(self.left_col)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(12)
        self.left_col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Rechte Spalte
        right_col = QWidget()
        right_col.setStyleSheet("QLabel { color: #FFFFFF; font-size: 25px; font-family: Cambria; }")
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.main_layout.addWidget(self.left_col)
        self.main_layout.addWidget(right_col)
        self.main_layout.setStretch(0, 3)  # 30%
        self.main_layout.setStretch(1, 7)  # 70%

        # Logo oben links
        logo_container = QWidget()
        logo_layout = QHBoxLayout(logo_container)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        logo_label = QLabel()
        logo_pixmap = QPixmap(os.path.join("logos", "PicMe_logo_cleaned.png"))
        if not logo_pixmap.isNull():
            logo_label.setPixmap(logo_pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            logo_label.setText("PicMe")
        logo_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        logo_layout.addWidget(logo_label)
        self.left_layout.addWidget(logo_container)

        # Erste Drop-Box (startet klein; wächst erst nach Bild-Drop)
        self.box1 = DragDropLabel("Drop image here")
        self.left_layout.addWidget(self.box1)

        # Toggle-Button (zweite Box)
        self.toggle_btn = QToolButton()
        self.toggle_btn.setObjectName("ToggleBtn")
        self.toggle_btn.setText("+")
        self.toggle_btn.setAutoRaise(False)
        self.toggle_btn.setFixedSize(28, 28)
        self.toggle_btn.clicked.connect(self.toggle_second_box)
        self.left_layout.addWidget(self.toggle_btn)

        # Platz für zweite Box
        self.second_box = None
        self.left_layout.addStretch()

        # Rechte Seite Placeholder
        right_placeholder = QLabel("Best fitting images")
        right_placeholder.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(right_placeholder)

        # 30%-Max-Breite links, aber: vor erstem Drop KEINE Höhenänderung
        self._apply_left_max_width()

    def toggle_second_box(self):
        if self.second_box is None:
            self.second_box = DragDropLabel("Add second image")
            idx_box1 = self.left_layout.indexOf(self.box1)
            self.left_layout.insertWidget(idx_box1 + 1, self.second_box)
            self.toggle_btn.setText("−")
            self.toggle_btn.setToolTip("Zweites Rechteck entfernen")
        else:
            self.left_layout.removeWidget(self.second_box)
            self.second_box.deleteLater()
            self.second_box = None
            self.toggle_btn.setText("+")
            self.toggle_btn.setToolTip("Weiteres Rechteck hinzufügen")

        self.left_layout.removeWidget(self.toggle_btn)
        last_box = self.second_box if self.second_box is not None else self.box1
        last_idx = self.left_layout.indexOf(last_box)
        self.left_layout.insertWidget(last_idx + 1, self.toggle_btn)

        self._apply_left_max_width()

    def resizeEvent(self, e: QResizeEvent):
        super().resizeEvent(e)
        self._apply_left_max_width()   # Fenstergröße bleibt; Boxbreite wird nur gedeckelt

    # Box-Breite links max. 30% Fensterbreite; KEINE Höhenänderung vor dem ersten Bild
    def _apply_left_max_width(self):
        outer_margin = 16
        max_w = max(200, int(self.width() * 0.30) - 2 * outer_margin)
        for box in (self.box1, self.second_box):
            if box is not None:
                box.setMaximumWidth(max_w)
                if box._orig_pm is not None:
                    # Nur wenn ein Bild vorhanden ist, Höhe neu fitten und Pixmap updaten
                    box._fit_height_to_image()
                    box._update_scaled_pixmap()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(os.path.join("logos", "PicMe_logo.png")))
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())
