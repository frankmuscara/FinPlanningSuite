from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSpinBox, QComboBox,
    QTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from ...core.store import new_client, save_client, list_clients, load_client, export_clients_csv, Client
from ...ui.event_bus import set_current_client


ACCOUNT_TYPES = {
    "Taxable Brokerage": "Taxable",
    "Traditional IRA": "Tax-Deferred",
    "Roth IRA": "Tax-Free",
    "401(k)": "Tax-Deferred",
    "403(b)": "Tax-Deferred",
    "SEP IRA": "Tax-Deferred",
    "Inherited IRA (Traditional)": "Tax-Deferred",
    "Inherited IRA (Roth)": "Tax-Free",
    "529 Plan": "Tax-Advantaged (Education)",
    "HSA": "Tax-Advantaged (Health)",
    "Other": "Custom",
}

class ClientProfileView(QWidget):
    def __init__(self):
        super().__init__()
        self.current: Client = new_client()

        root = QVBoxLayout(self)

        title = QLabel("Client Profile")
        title.setStyleSheet("font-size:20px; font-weight:600;")
        root.addWidget(title)

        # --- Top: client selector + actions ---
        top = QHBoxLayout()
        self.clientPicker = QComboBox()
        self.refresh_client_list()
        self.clientPicker.currentIndexChanged.connect(self.on_pick)
        top.addWidget(QLabel("Client:"))
        top.addWidget(self.clientPicker, 1)

        self.btn_new = QPushButton("New")
        self.btn_new.clicked.connect(self.on_new)
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self.on_save)
        self.btn_export = QPushButton("Export CSV (Access-friendly)")
        self.btn_export.clicked.connect(self.on_export)
        top.addStretch(1)
        top.addWidget(self.btn_new); top.addWidget(self.btn_save); top.addWidget(self.btn_export)
        root.addLayout(top)

        # --- Basic fields ---
        row = QHBoxLayout()
        self.fn = QLineEdit(); self.ln = QLineEdit()
        self.email = QLineEdit(); self.phone = QLineEdit()
        self.birth = QSpinBox(); self.birth.setRange(1900, 2100); self.birth.setValue(1985)
        self.spouse = QLineEdit(); self.spouse_birth = QSpinBox(); self.spouse_birth.setRange(1900, 2100)
        self.filing = QComboBox(); self.filing.addItems(["Single","Married Filing Jointly","Married Filing Separately","Head of Household"])
        self.state = QLineEdit("PA")

        col1 = QVBoxLayout(); col2 = QVBoxLayout(); col3 = QVBoxLayout()
        def add(col, lab, w):
            col.addWidget(QLabel(lab)); col.addWidget(w)

        add(col1, "First Name", self.fn); add(col1, "Last Name", self.ln); add(col1, "State", self.state)
        add(col2, "Email", self.email); add(col2, "Phone", self.phone); add(col2, "Filing Status", self.filing)
        add(col3, "Birth Year", self.birth); add(col3, "Spouse Name", self.spouse); add(col3, "Spouse Birth Year", self.spouse_birth)

        row.addLayout(col1,1); row.addLayout(col2,1); row.addLayout(col3,1)
        root.addLayout(row)

        # --- Planning inputs ---
        row2 = QHBoxLayout()
        self.ret_age = QSpinBox(); self.ret_age.setRange(40, 80); self.ret_age.setValue(65)
        self.ret_spend = QLineEdit()
        row2.addWidget(QLabel("Retirement Age")); row2.addWidget(self.ret_age)
        row2.addWidget(QLabel("Retirement Spending (today $)")); row2.addWidget(self.ret_spend,1)
        root.addLayout(row2)

        # --- Accounts table (very simple) ---
        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["Account Name", "Type", "Tax Status", "Value"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        btns = QHBoxLayout()
        self.btn_add_acct = QPushButton("Add Account")
        self.btn_del_acct = QPushButton("Delete Selected")
        self.btn_add_acct.clicked.connect(self.add_acct)
        self.btn_del_acct.clicked.connect(self.del_acct)
        btns.addWidget(self.btn_add_acct); btns.addWidget(self.btn_del_acct); btns.addStretch(1)

        root.addWidget(QLabel("Accounts"))
        root.addWidget(self.tbl, 1); root.addLayout(btns)

        # --- Notes ---
        self.notes = QTextEdit()
        root.addWidget(QLabel("Notes"))
        root.addWidget(self.notes, 1)

        self.load_into_form(self.current)

    # ---- helpers ----
    def refresh_client_list(self):
        self.clientPicker.blockSignals(True)
        self.clientPicker.clear()
        self._clients = list_clients()
        for c in self._clients:
            self.clientPicker.addItem(f"{c.last_name}, {c.first_name}  ({c.client_id[:8]})", c.client_id)
        self.clientPicker.blockSignals(False)

    def on_pick(self, idx: int):
        if idx < 0 or idx >= len(self._clients): return
        cid = self.clientPicker.currentData()
        c = load_client(cid)
        if c:
            self.current = c
            self.load_into_form(c)
        set_current_client(self.current)

    def on_new(self):
        self.current = new_client()
        self.load_into_form(self.current)
        self.clientPicker.setCurrentIndex(-1)
        set_current_client(self.current)

    def on_save(self):
            c = self.collect_from_form()
            if c is None:
                QMessageBox.critical(self, "Save failed", "Could not assemble client record from the form.")
                return
            save_client(c)
            # reload from disk so UI reflects persisted object
            self.current = load_client(c.client_id) or c
            self.load_into_form(self.current)
            self.refresh_client_list()
            QMessageBox.information(self, "Saved", "Client saved to data/clients/.")
            set_current_client(self.current)

    def on_export(self):
        export_clients_csv()
        QMessageBox.information(self, "Exported", "Exported all clients to data/clients_export.csv (import into Access).")

    def add_acct(self):
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)

        # Column 0: Account Name (editable text)
        self.tbl.setItem(r, 0, QTableWidgetItem(""))

        # Column 1: Type (combo)
        cb = self._make_type_combo(r)
        self.tbl.setCellWidget(r, 1, cb)

        # Column 2: Tax Status (auto-filled, not user-editable)
        self._set_tax_status_for_row(r, cb.currentText())

        # Column 3: Value (editable text)
        self.tbl.setItem(r, 3, QTableWidgetItem(""))


    def del_acct(self):
        r = self.tbl.currentRow()
        if r >= 0: self.tbl.removeRow(r)

    def update_tax_status(self, acct_type):
            status = ACCOUNT_TYPES.get(acct_type, "Unknown")
            self.tax_status_lbl.setText(status)
            self.client_data["tax_status"] = status  # if you’re storing in dict/object
    def _make_type_combo(self, row: int) -> QComboBox:
        cb = QComboBox(self.tbl)
        cb.addItems(ACCOUNT_TYPES.keys())
        cb.currentTextChanged.connect(lambda t, r=row: self._set_tax_status_for_row(r, t))
        return cb

    def _set_tax_status_for_row(self, row: int, acct_type: str):
        status = ACCOUNT_TYPES.get(acct_type, "Unknown")
        self.tbl.setItem(row, 2, QTableWidgetItem(status))



    # ---- form <-> model ----
    def load_into_form(self, c: Client):
        self.fn.setText(c.first_name or ""); self.ln.setText(c.last_name or "")
        self.email.setText(c.email or ""); self.phone.setText(c.phone or "")
        self.birth.setValue(c.birth_year or 1985)
        self.spouse.setText(c.spouse_name or ""); self.spouse_birth.setValue(c.spouse_birth_year or 1985)
        self.filing.setCurrentText(c.filing_status or "Married Filing Jointly")
        self.state.setText(c.state or "PA")
        self.ret_age.setValue(c.retirement_age or 65)
        self.ret_spend.setText("" if c.retirement_spending is None else f"{c.retirement_spending:.0f}")
        self.notes.setPlainText(c.notes or "")
        self.tbl.setRowCount(0)
        for a in (c.accounts or []):
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)

            # name
            self.tbl.setItem(r, 0, QTableWidgetItem(a.get("name","")))

            # type (combo)
            cb = self._make_type_combo(r)
            t = a.get("type","Taxable Brokerage")
            idx = cb.findText(t) if t else -1
            if idx >= 0: cb.setCurrentIndex(idx)
            self.tbl.setCellWidget(r, 1, cb)

            # tax status (derived; but load saved value if present)
            saved_tax = a.get("tax","")
            if not saved_tax:
                saved_tax = ACCOUNT_TYPES.get(t, "Unknown")
            self.tbl.setItem(r, 2, QTableWidgetItem(saved_tax))

            # value
            val = a.get("value","")
            self.tbl.setItem(r, 3, QTableWidgetItem(str(val)))


    def collect_from_form(self) -> Client | None:
        try:
            c = self.current  # should be a Client from new_client() or load_client()

            # ----- simple fields -----
            c.first_name = self.fn.text().strip()
            c.last_name  = self.ln.text().strip()
            c.email = self.email.text().strip()
            c.phone = self.phone.text().strip()
            c.birth_year = int(self.birth.value()) if self.birth.value() else None
            c.spouse_name = self.spouse.text().strip()
            c.spouse_birth_year = int(self.spouse_birth.value()) if self.spouse_birth.value() else None
            c.filing_status = self.filing.currentText()
            c.state = self.state.text().strip()
            c.retirement_age = int(self.ret_age.value()) if self.ret_age.value() else None

            txt = self.ret_spend.text().strip().replace(",", "")
            c.retirement_spending = float(txt) if txt else None

            # ----- accounts from table -----
            accts = []
            for r in range(self.tbl.rowCount()):
                name = self.tbl.item(r,0).text().strip() if self.tbl.item(r,0) else ""

                w = self.tbl.cellWidget(r,1)
                acct_type = w.currentText().strip() if isinstance(w, QComboBox) else (
                    self.tbl.item(r,1).text().strip() if self.tbl.item(r,1) else ""
                )

                tax = self.tbl.item(r,2).text().strip() if self.tbl.item(r,2) else ACCOUNT_TYPES.get(acct_type, "Unknown")

                val_txt = self.tbl.item(r,3).text().strip() if self.tbl.item(r,3) else "0"
                try:
                    value = float(val_txt.replace(",", "")) if val_txt else 0.0
                except ValueError:
                    value = 0.0

                accts.append({"name": name, "type": acct_type, "tax": tax, "value": value})

            c.accounts = accts
            c.notes = self.notes.toPlainText()

            return c

        except Exception as e:
            # Optional: log to console to see the real cause
            print(f"[collect_from_form] error: {e}")
            return None

