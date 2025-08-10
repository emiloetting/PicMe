from ColorSimilarity.main_helper import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import os



# GUI-MODULE TO HANDLE USER INTERACTIONS AND DISPLAY RESULTS----------------------------------------------------------------------

class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_GUI()
        # Add widgets and layout here

    def init_GUI(self):
        # -------INIT GUI WINDOW---------------------------------------
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

        # Make title bar in desired color
        self.setStyleSheet("QMainWindow { background-color: #011126; }")
        self.setGeometry(x_position, y_position, window_w, window_h)


        # --------DEFINE MAIN GUI COMPONENTS-----------------------------
        # Create main window layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Define left and right vertikal layouts
        left_col = QWidget()
        self.left_layout = QVBoxLayout(left_col)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(12)

        right_col = QWidget()
        right_col.setStyleSheet("QLabel { color: #FFFFFF; font-size: 25px; font-family: Cambria; }")
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)

        main_layout.addWidget(left_col)  
        main_layout.addWidget(right_col, 1)  # Add both to main layout   

        # create widget for logo
        logo_container = QWidget()
        logo_layout = QHBoxLayout(logo_container)

        # Insert logo
        logo_label = QLabel()
        logo_pixmap = QPixmap(os.path.join("logos", "PicMe_logo_cleaned.png"))
        scaled_logo = logo_pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(scaled_logo)
        logo_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # Add logo to layout
        logo_layout.addWidget(logo_label)

        # Add logo container to LEFT layout
        self.left_layout.addWidget(logo_container)
        self.left_layout.addStretch()  # Keeps logo in place

        # PLACEHOLDER
        right_placeholder = QLabel("Best fitting images")
        right_placeholder.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(right_placeholder)






if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(os.path.join("logos", "PicMe_logo.png")))
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())