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
from ..core.store import list_clients
from .event_bus import set_current_client

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
    def __init__(self):
        super().__init__()
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

        try:
            clients = list_clients()
            if clients:
                set_current_client(clients[0])  # newest-first from your store
        except Exception:
            pass

def launch_app():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
