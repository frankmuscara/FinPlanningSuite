# scaffold.py  —  generates a PyQt6 shell app with all views + requirements.txt

import os

BASE = r"C:\Users\FrankMuscara\GCFP"

PKG = os.path.join(BASE, "finplan_suite")
VIEWS = os.path.join(PKG, "ui", "views")
CORE = os.path.join(PKG, "core")
CONFIG = os.path.join(PKG, "config")
DATA = os.path.join(PKG, "data")
TESTS = os.path.join(PKG, "tests")

for p in [PKG, VIEWS, CORE, CONFIG, DATA, TESTS]:
    os.makedirs(p, exist_ok=True)

# ---------- requirements.txt ----------
requirements_txt = """\
PyQt6>=6.6
numpy>=1.26
pandas>=2.0
scipy>=1.11
cvxpy>=1.4
statsmodels>=0.14
plotly>=5.20
matplotlib>=3.8
PyYAML>=6.0
"""

# ---------- README.md ----------
readme_md = """\
# FinPlan Suite (Shell Prototype)

A PyQt6 desktop shell for a financial planning suite with module stubs:
- Dashboard
- Client Profile
- Economic Forecasts
- Capital Market Assumptions
- Risk Tolerance
- Portfolio Builder
- Monte Carlo & Cashflows
- Tax & Estate
- Reports

## Quick start
```powershell
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r requirements.txt
python run.py
"""

#---------- run.py ----------
run_py = """
from finplan_suite.app import launch_app

if name == "main":
launch_app()
"""

#---------- package files ----------
init_py = "" # empty init.py

app_py = """
from .ui.main_window import launch_app

if name == "main":
launch_app()
"""

main_window_py = """\
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QListWidget,
    QListWidgetItem, QStackedWidget, QToolBar, QPushButton, QLineEdit
)
from PyQt6.QtCore import Qt
import sys

from .views.dashboard_view import DashboardView
from .views.client_profile_view import ClientProfileView
from .views.economic_view import EconomicView
from .views.cma_view import CMAView
from .views.risk_view import RiskView
from .views.portfolio_view import PortfolioView
from .views.monte_view import MonteCarloView
from .views.estate_view import EstateView
from .views.reports_view import ReportsView

SECTIONS = [
    ("Dashboard", DashboardView),
    ("Client Profile", ClientProfileView),
    ("Economic Forecasts", EconomicView),
    ("Capital Market Assumptions", CMAView),
    ("Risk Tolerance", RiskView),
    ("Portfolio Builder", PortfolioView),
    ("Monte Carlo & Cashflows", MonteCarloView),
    ("Tax & Estate", EstateView),
    ("Reports", ReportsView),
]

class MainWindow(QMainWindow):
    def init(self):
        super().init()
        self.setWindowTitle("FinPlan Suite — Advisor Prototype")
        self.resize(1200, 800)

        # Top toolbar
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.clientSearch = QLineEdit()
        self.clientSearch.setPlaceholderText("Search clients…")
        self.clientSearch.setFixedWidth(240)
        toolbar.addWidget(self.clientSearch)
        toolbar.addSeparator()

        self.scenarioBtn = QPushButton("Scenario: Base")
        toolbar.addWidget(self.scenarioBtn)
        toolbar.addSeparator()

        self.saveBtn = QPushButton("Save")
        self.exportBtn = QPushButton("Export")
        toolbar.addWidget(self.saveBtn)
        toolbar.addWidget(self.exportBtn)

        # Central layout
        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)

        # Sidebar
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(240)
        self.sidebar.setStyleSheet("QListWidget { border-right: 1px solid #ddd; }")
        for name, _ in SECTIONS:
        self.sidebar.addItem(QListWidgetItem(name))

        # Stack
        self.stack = QStackedWidget()
        self.views = []
        for _, view_cls in SECTIONS:
            view = view_cls()
            self.views.append(view)
            self.stack.addWidget(view)

    self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
    self.sidebar.setCurrentRow(0)

    outer.addWidget(self.sidebar)
    outer.addWidget(self.stack, 1)
    self.setCentralWidget(central)

def launch_app():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
"""

#---------- stub views ----------
STUB = """\
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class {ClassName}(QWidget):
    def init(self):
        super().init()
        layout = QVBoxLayout(self)
        title = QLabel("{Title}")
        title.setStyleSheet("font-size:20px; font-weight:600;")
        layout.addWidget(title)
        hint = QLabel("{Hint}")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#555;")
        layout.addWidget(hint)
        layout.addStretch(1)
"""

view_defs = {
    "dashboard_view.py": ("DashboardView", "Dashboard", "Plan snapshot, portfolio snapshot, to-dos, scenario tiles."),
    "client_profile_view.py": ("ClientProfileView", "Client Profile", "Household data, accounts, goals, jurisdiction."),
    "economic_view.py": ("EconomicView", "Economic Forecasts", "Macro models, fan charts, overrides, scenario save."),
    "cma_view.py": ("CMAView", "Capital Market Assumptions", "Expected returns, volatility, correlation heatmap."),
    "risk_view.py": ("RiskView", "Risk Tolerance", "Questionnaire, score → band, policy constraints."),
    "portfolio_view.py": ("PortfolioView", "Portfolio Builder", "Efficient frontier, constraints, allocation sliders."),
    "monte_view.py": ("MonteCarloView", "Monte Carlo & Cashflows", "Cashflow editor, trials, probability of success."),
    "estate_view.py": ("EstateView", "Tax & Estate", "Net-to-heirs waterfall, sensitivities, VUL vs BTID."),
    "reports_view.py": ("ReportsView", "Reports", "Generate white-labeled PDFs/CSVs with assumptions appendix."),
}

#---------- core placeholders ----------
core_files = {
    "forecasting.py": "# TODO: ARIMA/VAR/Bayesian macro forecasting engine\n",
    "cma.py": "# TODO: Translate macro to asset class CMAs\n",
    "risk_tolerance.py": "# TODO: Questionnaire scoring + constraints mapping\n",
    "portfolio.py": "# TODO: Mean-variance optimizer with cvxpy\n",
    "monte_carlo.py": "# TODO: MC simulator with cashflow modeling\n",
    "tax_estate.py": "# TODO: Estate/tax sensitivity engine\n",
}

#---------- write files ----------
def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# ---------- write everything ----------
write_file(os.path.join(BASE, "requirements.txt"), requirements_txt)
write_file(os.path.join(BASE, "README.md"), readme_md)
write_file(os.path.join(BASE, "run.py"), run_py)

write_file(os.path.join(PKG, "__init__.py"), init_py)
write_file(os.path.join(PKG, "app.py"), app_py)
write_file(os.path.join(PKG, "ui", "main_window.py"), main_window_py)

for fname, (cls, title, hint) in view_defs.items():
    write_file(os.path.join(VIEWS, fname), STUB_VIEW.format(ClassName=cls, Title=title, Hint=hint))

for fname, content in core_files.items():
    write_file(os.path.join(CORE, fname), content)

print("✅ Scaffold complete.\nNext:")
print(f"  1) cd {BASE}")
print(r"  2) python -m venv .venv")
print(r"  3) .\.venv\Scripts\activate")
print(r"  4) pip install -r requirements.txt")
print(r"  5) python run.py")

# Open PowerShell and run it (yourself dummy)