from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class EstateView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("Tax & Estate")
        title.setStyleSheet("font-size:20px; font-weight:600;")
        layout.addWidget(title)
        hint = QLabel("Net-to-heirs waterfall, sensitivities, VUL vs BTID.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#555;")
        layout.addWidget(hint)
        layout.addStretch(1)
