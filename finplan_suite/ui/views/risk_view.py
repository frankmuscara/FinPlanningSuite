from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QPushButton, QHBoxLayout, QMessageBox, QComboBox
from PyQt6.QtCore import Qt, QEvent
from ...core.store import save_client, list_clients, load_client, Client

QUESTIONS = [
    "How would you react to a 15% decline in your portfolio over 3 months?",
    "What is your primary goal for this portfolio?",
    "How many years until you expect to draw from this portfolio?",
    "How stable is your income and emergency fund?",
    "How experienced are you with market volatility?",
]
CHOICES = [
    ["Sell to avoid further loss (1)","Wait it out (3)","Buy more (5)"],
    ["Capital preservation (1)","Balanced growth (3)","Aggressive growth (5)"],
    ["< 3 years (1)","3–10 years (3)","> 10 years (5)"],
    ["Low/limited (1)","Moderate (3)","High/ample (5)"],
    ["New (1)","Some (3)","Experienced (5)"],
]

def score_to_band(score: int) -> str:
    if score <= 9: return "Conservative"
    if score <= 13: return "Moderately Conservative"
    if score <= 17: return "Moderate"
    if score <= 21: return "Moderately Aggressive"
    return "Aggressive"

class RiskView(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)

        title = QLabel("Risk Tolerance")
        title.setStyleSheet("font-size:20px; font-weight:600;")
        root.addWidget(title)

        # pick client
        row = QHBoxLayout()
        self.clientPicker = QComboBox(); self._clients=[]
        self.refresh_clients()
        row.addWidget(QLabel("Client:")); row.addWidget(self.clientPicker,1)
        root.addLayout(row)

        self.btn_refresh = QPushButton("Reload Clients")
        self.btn_refresh.clicked.connect(self.refresh_clients)
        row.addWidget(self.btn_refresh)

        self.tbl = QTableWidget(len(QUESTIONS), 2)
        self.tbl.setHorizontalHeaderLabels(["Question", "Response"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        for i, q in enumerate(QUESTIONS):
            self.tbl.setItem(i, 0, QTableWidgetItem(q))
            cb = QComboBox(); cb.addItems(CHOICES[i])
            self.tbl.setCellWidget(i, 1, cb)

        root.addWidget(self.tbl)

        btns = QHBoxLayout()
        self.btn_score = QPushButton("Score & Save to Client")
        self.btn_score.clicked.connect(self.on_score)
        btns.addStretch(1); btns.addWidget(self.btn_score)
        root.addLayout(btns)

        self.result = QLabel("")
        self.result.setStyleSheet("font-weight:600;")
        root.addWidget(self.result)

    def refresh_clients(self):
        self.clientPicker.blockSignals(True)
        self.clientPicker.clear()
        self._clients = list_clients()
        for c in self._clients:
            self.clientPicker.addItem(f"{c.last_name}, {c.first_name}", c.client_id)
        self.clientPicker.blockSignals(False)

    def on_score(self):
        if not self._clients or self.clientPicker.currentIndex()<0:
            QMessageBox.warning(self, "No client", "Select or create a client in Client Profile first.")
            return
        score = 0
        for i in range(self.tbl.rowCount()):
            cb = self.tbl.cellWidget(i,1)
            sel = cb.currentText()
            score += int(sel.split("(")[-1].rstrip(")"))
        band = score_to_band(score)
        cid = self.clientPicker.currentData()
        c = load_client(cid)
        if c:
            c.risk_score = score
            c.risk_band = band
            save_client(c)
        self.result.setText(f"Risk Score: {score} → {band}")
        QMessageBox.information(self, "Saved", f"Saved risk score ({score}) and band ({band}) to client.")
    def showEvent(self, event: QEvent):
        super().showEvent(event)
        # Each time the tab is shown, reload the client list
        self.refresh_clients()
    def refresh_clients(self):
        current_id = self.clientPicker.currentData()
        self.clientPicker.blockSignals(True)
        self.clientPicker.clear()
        self._clients = list_clients()
        for c in self._clients:
            self.clientPicker.addItem(f"{c.last_name}, {c.first_name}", c.client_id)
        self.clientPicker.blockSignals(False)

    # Try to keep prior selection; else select first item if available
        if self._clients:
         if current_id:
            idx = next((i for i, c in enumerate(self._clients) if c.client_id == current_id), 0)
            self.clientPicker.setCurrentIndex(idx)
        else:
            self.clientPicker.setCurrentIndex(0)
