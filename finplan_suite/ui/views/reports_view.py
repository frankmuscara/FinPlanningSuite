from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class ReportsView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("Reports")
        title.setStyleSheet("font-size:20px; font-weight:600;")
        layout.addWidget(title)
        hint = QLabel("Generate white-labeled PDFs/CSVs with assumptions appendix.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#555;")
        layout.addWidget(hint)
        layout.addStretch(1)
