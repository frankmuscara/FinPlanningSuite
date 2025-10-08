from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class DashboardView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("Dashboard")
        title.setStyleSheet("font-size:20px; font-weight:600;")
        layout.addWidget(title)
        hint = QLabel("Plan snapshot, portfolio snapshot, to-dos, scenario tiles.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#555;")
        layout.addWidget(hint)
        layout.addStretch(1)
