from ColorSimilarity.main_helper import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import os


class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_GUI()

    def init_GUI(self):
        # ------- INIT WINDOW -------
        screen_rect = QApplication.desktop().screenGeometry()
        screen_width = screen_rect.width()
        screen_height = screen_rect.height()

        window_w = int(screen_width * 0.75)
        window_h = int(screen_height * 0.75)
        x_position = (screen_width - window_w) // 2
        y_position = (screen_height - window_h) // 2

        self.setWindowTitle("PicMe - Pixel-Informed Content-Matching Engine")
        self.setGeometry(x_position, y_position, window_w, window_h)
        self.setWindowIcon(QIcon(os.path.join("logos", "PicMe_logo.png")))

        # Styles (inkl. Drag-Highlight)
        self.setStyleSheet("""
        QMainWindow { background-color: #011126; }

        QFrame#DashedBox {
            border: 2px dashed #9aa0a6;
            border-radius: 8px;
            background: rgba(255,255,255,0.03);
        }
        QFrame#DashedBox[dragActive="true"] {
            border-color: #4aa3ff;
            background: rgba(74,163,255,0.08);
        }
        #DashedBox QLabel {
            color: #E6EDF3;
            font-size: 12pt;
            qproperty-alignment: AlignCenter;
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

        # ------- MAIN LAYOUT -------
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Left column
        left_col = QWidget()
        self.left_layout = QVBoxLayout(left_col)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(12)
        left_col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Right column
        right_col = QWidget()
        right_col.setStyleSheet("QLabel { color: #FFFFFF; font-size: 25px; font-family: Cambria; }")
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        main_layout.addWidget(left_col)
        main_layout.addWidget(right_col)
        main_layout.setStretch(0, 3)  # 30%
        main_layout.setStretch(1, 7)  # 70%

        # ------- Logo (left, top) -------
        logo_container = QWidget()
        logo_layout = QHBoxLayout(logo_container)
        logo_layout.setContentsMargins(0, 0, 0, 0)

        logo_label = QLabel()
        logo_pixmap = QPixmap(os.path.join("logos", "PicMe_logo_cleaned.png"))
        if not logo_pixmap.isNull():
            scaled_logo = logo_pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_logo)
        else:
            logo_label.setText("PicMe")
        logo_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        logo_layout.addWidget(logo_label)

        # Add logo to left column and remember it
        self.logo_container = logo_container
        self.left_layout.addWidget(self.logo_container)

        # ------- Box 1 directly AFTER logo -------
        self.box1 = self.create_dashed_box("Drop image here")
        insert_after_logo = self.left_layout.indexOf(self.logo_container) + 1
        self.left_layout.insertWidget(insert_after_logo, self.box1)
        self._install_drop(self.box1)  # Drag&Drop aktivieren

        # Toggle button UNDER the boxes (starts as "+")
        self.toggle_btn = QToolButton()
        self.toggle_btn.setObjectName("ToggleBtn")
        self.toggle_btn.setText("+")
        self.toggle_btn.setAutoRaise(False)
        self.toggle_btn.setFixedSize(28, 28)
        self.toggle_btn.clicked.connect(self.toggle_second_box)
        self.left_layout.insertWidget(insert_after_logo + 1, self.toggle_btn)

        # Second box placeholder
        self.second_box = None

        # Stretch at the end
        self.left_layout.addStretch()

        # ------- Right placeholder -------
        right_placeholder = QLabel("Best fitting images")
        right_placeholder.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(right_placeholder)

    # ---------- Helpers ----------
    def create_dashed_box(self, text: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("DashedBox")
        frame.setAcceptDrops(True)               # Drop erlauben
        frame.setProperty("dragActive", False)   # fürs Highlight
        frame.setMinimumHeight(140)

        lay = QVBoxLayout(frame)
        lay.setContentsMargins(12, 12, 12, 12)

        label = QLabel(text)
        label.setObjectName("BoxLabel")
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        lay.addWidget(label, alignment=Qt.AlignCenter)
        return frame

    def toggle_second_box(self):
        if self.second_box is None:
            # Insert second box directly UNDER box1
            self.second_box = self.create_dashed_box("Add second image")
            idx_box1 = self.left_layout.indexOf(self.box1)
            self.left_layout.insertWidget(idx_box1 + 1, self.second_box)
            self._install_drop(self.second_box)  # Drag&Drop aktivieren

            # Button to "minus"
            self.toggle_btn.setText("−")
            self.toggle_btn.setToolTip("Zweites Rechteck entfernen")
        else:
            # Remove second box
            self.left_layout.removeWidget(self.second_box)
            self.second_box.deleteLater()
            self.second_box = None

            # Button back to "plus"
            self.toggle_btn.setText("+")
            self.toggle_btn.setToolTip("Weiteres Rechteck hinzufügen")

        # Always move the button directly UNDER the last existing box
        self.left_layout.removeWidget(self.toggle_btn)
        last_box = self.second_box if self.second_box is not None else self.box1
        last_idx = self.left_layout.indexOf(last_box)
        self.left_layout.insertWidget(last_idx + 1, self.toggle_btn)

    # --- Drag & Drop minimal via Event-Filter + Auto-Resize ---
    def _install_drop(self, frame: QFrame):
        frame.installEventFilter(self)

    def eventFilter(self, obj, event):
        if isinstance(obj, QFrame) and obj.objectName() == "DashedBox":
            et = event.type()
            if et == QEvent.DragEnter:
                if self._has_acceptable_data(event.mimeData()):
                    event.acceptProposedAction()
                    self._set_drag_active(obj, True)
                    return True
            elif et == QEvent.DragMove:
                if self._has_acceptable_data(event.mimeData()):
                    event.acceptProposedAction()
                    return True
            elif et == QEvent.DragLeave:
                self._set_drag_active(obj, False)
                return True
            elif et == QEvent.Drop:
                self._set_drag_active(obj, False)
                path = self._first_local_image_path(event.mimeData())
                if path:
                    self._set_box_pixmap(obj, path)
                    event.acceptProposedAction()
                    return True
            elif et == QEvent.Resize:
                # Auto-Resize der Vorschau
                self._rescale_to_frame(obj)
                return False
        return super().eventFilter(obj, event)

    def _has_acceptable_data(self, md: QMimeData) -> bool:
        if md.hasUrls():
            return self._first_local_image_path(md) is not None
        return md.hasImage()

    def _first_local_image_path(self, md: QMimeData):
        if not md.hasUrls():
            return None
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
        for url in md.urls():
            if url.isLocalFile():
                p = url.toLocalFile()
                _, e = os.path.splitext(p)
                if e.lower() in exts and os.path.isfile(p):
                    return p
        return None

    def _set_drag_active(self, frame: QFrame, active: bool):
        if frame.property("dragActive") != active:
            frame.setProperty("dragActive", active)
            frame.style().unpolish(frame)
            frame.style().polish(frame)
            frame.update()

    def _set_box_pixmap(self, frame: QFrame, path: str):
        pm = QPixmap(path)
        if pm.isNull():
            return
        # Original-Pixmap am Frame merken (für spätere Resizes)
        frame._orig_pm = pm
        self._rescale_to_frame(frame)
        # Text verstecken
        label = frame.findChild(QLabel, "BoxLabel")
        if label:
            label.setText("")

    def _rescale_to_frame(self, frame: QFrame):
        pm_orig = getattr(frame, "_orig_pm", None)
        if pm_orig is None:
            return
        label = frame.findChild(QLabel, "BoxLabel")
        if not label:
            return
        lay = frame.layout()
        if lay is not None:
            m = lay.contentsMargins()
            avail_w = max(1, frame.width()  - (m.left() + m.right()))
            avail_h = max(1, frame.height() - (m.top()  + m.bottom()))
        else:
            avail_w, avail_h = max(1, frame.width()-24), max(1, frame.height()-24)
        target = QSize(avail_w, avail_h)
        pm_scaled = pm_orig.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pm_scaled)

class ThumbLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._orig = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(120, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setPixmap(self, pm: QPixmap):
        self._orig = pm
        super().setPixmap(self._scaled())

    def resizeEvent(self, e: QResizeEvent):
        if self._orig is not None:
            super().setPixmap(self._scaled())
        super().resizeEvent(e)

    def _scaled(self):
        if self._orig is None or self.width() <= 0 or self.height() <= 0:
            return self._orig
        return self._orig.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(os.path.join("logos", "PicMe_logo.png")))
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())
