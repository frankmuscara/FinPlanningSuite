# finplan_suite/ui/views/portfolio_view.py
"""Portfolio Builder view with model-based portfolio construction."""

import json
import os
from typing import Dict, List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QFrame, QGridLayout, QMessageBox, QTabWidget,
    QComboBox, QHeaderView, QGroupBox, QProgressBar, QSplitter,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ...core.cma import load_cma_json, CMA
from ...core.portfolio import build_frontier
from .hammer_view import HammerView


# Path to models config
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data")
MODELS_FILE = os.path.join(DATA_DIR, "models.json")


def load_models() -> Dict[str, dict]:
    """Load portfolio models from JSON config."""
    if not os.path.exists(MODELS_FILE):
        return {}
    try:
        with open(MODELS_FILE, "r") as f:
            data = json.load(f)
        return data.get("models", {})
    except Exception:
        return {}


def save_models(models: Dict[str, dict]):
    """Save portfolio models to JSON config."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MODELS_FILE, "w") as f:
        json.dump({"models": models}, f, indent=2)


class TickerFetchWorker(QThread):
    """Worker thread for fetching ticker data."""

    finished = pyqtSignal(dict)  # ticker -> TickerInfo
    progress = pyqtSignal(str, int, int)  # message, current, total
    error = pyqtSignal(str)

    def __init__(self, tickers: List[str]):
        super().__init__()
        self.tickers = tickers

    def run(self):
        try:
            from ...core.ticker_data import fetch_ticker_info

            results = {}
            total = len(self.tickers)

            for i, ticker in enumerate(self.tickers):
                self.progress.emit(f"Fetching {ticker}...", i + 1, total)
                results[ticker.upper()] = fetch_ticker_info(ticker)

            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class PortfolioView(QWidget):
    """Portfolio Builder with model-based construction and HAMMER integration."""

    def __init__(self):
        super().__init__()
        self.setObjectName("PortfolioView")

        # Data
        self.cma = self._load_cma_or_default()
        self.frontier_result = None
        self.models = load_models()
        self.ticker_data: Dict = {}
        self._fetch_worker: Optional[TickerFetchWorker] = None

        # Main layout with tabs
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 1: Portfolio Builder
        builder_tab = self._create_builder_tab()
        self.tabs.addTab(builder_tab, "Portfolio Builder")

        # Tab 2: HAMMER Analysis
        self.hammer_view = HammerView()
        self.tabs.addTab(self.hammer_view, "HAMMER Analysis")

        # Initial state
        self._compute_frontier()
        self._populate_model_dropdown()

    def _load_cma_or_default(self) -> CMA:
        """Load CMA or create default."""
        cma = load_cma_json(path=os.path.join(DATA_DIR, "cma.json"))
        if cma is None:
            from ...core.cma import derive_cma_from_macro
            cma = derive_cma_from_macro(gdp=0.017, cpi=0.025, real_short=0.01, term_premium=0.015)
        return cma

    def _create_builder_tab(self) -> QWidget:
        """Create the Portfolio Builder tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Header
        header = QLabel("Portfolio Builder")
        header.setStyleSheet("font-size: 20px; font-weight: 600;")
        layout.addWidget(header)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Frontier chart + Model selector
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)

        # Efficient Frontier chart (reference only)
        frontier_group = QGroupBox("Efficient Frontier (Reference)")
        frontier_layout = QVBoxLayout(frontier_group)

        self.fig = Figure(figsize=(5, 3.5), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        frontier_layout.addWidget(self.canvas)

        frontier_note = QLabel("Based on CMAs. For reference only - select models below.")
        frontier_note.setStyleSheet("color: #666; font-size: 11px;")
        frontier_layout.addWidget(frontier_note)

        left_layout.addWidget(frontier_group)

        # Model selector
        model_group = QGroupBox("Portfolio Model")
        model_layout = QVBoxLayout(model_group)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Select Model:"))
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self._on_model_selected)
        model_row.addWidget(self.model_combo, 1)
        model_layout.addLayout(model_row)

        self.model_desc_label = QLabel("")
        self.model_desc_label.setStyleSheet("color: #555; font-style: italic;")
        self.model_desc_label.setWordWrap(True)
        model_layout.addWidget(self.model_desc_label)

        left_layout.addWidget(model_group)
        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # Right panel: Holdings table + Stats
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 0, 0)

        # Holdings table
        holdings_group = QGroupBox("Holdings")
        holdings_layout = QVBoxLayout(holdings_group)

        self.holdings_table = QTableWidget()
        self.holdings_table.setColumnCount(5)
        self.holdings_table.setHorizontalHeaderLabels(["Ticker", "Weight %", "Name", "Expense Ratio", "Yield"])

        header = self.holdings_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        self.holdings_table.setAlternatingRowColors(True)
        self.holdings_table.cellChanged.connect(self._on_cell_changed)
        holdings_layout.addWidget(self.holdings_table)

        # Holdings buttons
        holdings_btn_row = QHBoxLayout()
        self.btn_add_row = QPushButton("Add Row")
        self.btn_add_row.clicked.connect(self._add_holding_row)
        holdings_btn_row.addWidget(self.btn_add_row)

        self.btn_remove_row = QPushButton("Remove Row")
        self.btn_remove_row.clicked.connect(self._remove_holding_row)
        holdings_btn_row.addWidget(self.btn_remove_row)

        holdings_btn_row.addStretch()

        self.weight_warning = QLabel("")
        self.weight_warning.setStyleSheet("color: #c00;")
        holdings_btn_row.addWidget(self.weight_warning)

        holdings_layout.addLayout(holdings_btn_row)
        right_layout.addWidget(holdings_group, 2)

        # Portfolio stats
        stats_group = QGroupBox("Portfolio Statistics")
        stats_layout = QGridLayout(stats_group)

        # Stats labels
        stats_layout.addWidget(QLabel("Weighted Expense Ratio:"), 0, 0)
        self.lbl_expense = QLabel("—")
        self.lbl_expense.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self.lbl_expense, 0, 1)

        stats_layout.addWidget(QLabel("Weighted Yield:"), 0, 2)
        self.lbl_yield = QLabel("—")
        self.lbl_yield.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self.lbl_yield, 0, 3)

        stats_layout.addWidget(QLabel("1-Year Return:"), 1, 0)
        self.lbl_return_1yr = QLabel("—")
        self.lbl_return_1yr.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self.lbl_return_1yr, 1, 1)

        stats_layout.addWidget(QLabel("5-Year Return:"), 1, 2)
        self.lbl_return_5yr = QLabel("—")
        self.lbl_return_5yr.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self.lbl_return_5yr, 1, 3)

        stats_layout.addWidget(QLabel("10-Year Return:"), 2, 0)
        self.lbl_return_10yr = QLabel("—")
        self.lbl_return_10yr.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self.lbl_return_10yr, 2, 1)

        right_layout.addWidget(stats_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #666;")
        right_layout.addWidget(self.progress_label)

        # Action buttons
        btn_row = QHBoxLayout()

        self.btn_validate = QPushButton("Validate && Fetch Data")
        self.btn_validate.setStyleSheet("padding: 8px 16px; font-weight: bold;")
        self.btn_validate.clicked.connect(self._validate_and_fetch)
        btn_row.addWidget(self.btn_validate)

        self.btn_send_hammer = QPushButton("Send to HAMMER")
        self.btn_send_hammer.clicked.connect(self._send_to_hammer)
        btn_row.addWidget(self.btn_send_hammer)

        self.btn_send_mc = QPushButton("Send to Monte Carlo")
        self.btn_send_mc.clicked.connect(self._send_to_monte_carlo)
        btn_row.addWidget(self.btn_send_mc)

        btn_row.addStretch()
        right_layout.addLayout(btn_row)

        splitter.addWidget(right_panel)

        # Set splitter sizes (35% left, 65% right)
        splitter.setSizes([350, 650])

        layout.addWidget(splitter)

        return tab

    def _populate_model_dropdown(self):
        """Populate the model selector dropdown."""
        self.model_combo.blockSignals(True)
        self.model_combo.clear()

        self.model_combo.addItem("Custom")
        for name in sorted(self.models.keys()):
            self.model_combo.addItem(name)

        self.model_combo.blockSignals(False)

    def _on_model_selected(self, model_name: str):
        """Handle model selection."""
        if model_name == "Custom":
            self.model_desc_label.setText("Create your own portfolio allocation.")
            # Clear table for custom entry
            self.holdings_table.setRowCount(0)
            self._add_holding_row()  # Start with one empty row
            return

        model = self.models.get(model_name)
        if not model:
            return

        # Update description
        desc = model.get("description", "")
        self.model_desc_label.setText(desc)

        # Populate holdings table
        holdings = model.get("holdings", {})
        self._populate_holdings_table(holdings)

        # Clear stats until validated
        self._clear_stats()

    def _populate_holdings_table(self, holdings: Dict[str, float]):
        """Populate the holdings table with given holdings."""
        self.holdings_table.blockSignals(True)
        self.holdings_table.setRowCount(0)

        for ticker, weight in sorted(holdings.items(), key=lambda x: -x[1]):
            row = self.holdings_table.rowCount()
            self.holdings_table.insertRow(row)

            # Ticker (editable)
            ticker_item = QTableWidgetItem(ticker)
            self.holdings_table.setItem(row, 0, ticker_item)

            # Weight (editable)
            weight_item = QTableWidgetItem(f"{weight * 100:.2f}")
            weight_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.holdings_table.setItem(row, 1, weight_item)

            # Name, Expense, Yield (populated on validation)
            for col in [2, 3, 4]:
                item = QTableWidgetItem("—")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.holdings_table.setItem(row, col, item)

        self.holdings_table.blockSignals(False)
        self._update_weight_warning()

    def _add_holding_row(self):
        """Add a new empty row to holdings table."""
        row = self.holdings_table.rowCount()
        self.holdings_table.insertRow(row)

        # Ticker (editable)
        self.holdings_table.setItem(row, 0, QTableWidgetItem(""))

        # Weight (editable)
        weight_item = QTableWidgetItem("0.00")
        weight_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.holdings_table.setItem(row, 1, weight_item)

        # Non-editable columns
        for col in [2, 3, 4]:
            item = QTableWidgetItem("—")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.holdings_table.setItem(row, col, item)

    def _remove_holding_row(self):
        """Remove selected row from holdings table."""
        current_row = self.holdings_table.currentRow()
        if current_row >= 0:
            self.holdings_table.removeRow(current_row)
            self._update_weight_warning()

    def _on_cell_changed(self, row: int, col: int):
        """Handle cell changes in holdings table."""
        if col == 1:  # Weight column
            self._update_weight_warning()

    def _update_weight_warning(self):
        """Update weight sum warning."""
        total = self._get_total_weight()
        if abs(total - 100.0) > 0.01:
            self.weight_warning.setText(f"Weights sum to {total:.2f}% (should be 100%)")
        else:
            self.weight_warning.setText("")

    def _get_total_weight(self) -> float:
        """Get total weight from table."""
        total = 0.0
        for row in range(self.holdings_table.rowCount()):
            item = self.holdings_table.item(row, 1)
            if item:
                try:
                    total += float(item.text())
                except ValueError:
                    pass
        return total

    def _get_holdings(self) -> Dict[str, float]:
        """Get holdings dict from table (weights as decimals 0-1)."""
        holdings = {}
        for row in range(self.holdings_table.rowCount()):
            ticker_item = self.holdings_table.item(row, 0)
            weight_item = self.holdings_table.item(row, 1)

            if ticker_item and weight_item:
                ticker = ticker_item.text().strip().upper()
                if ticker:
                    try:
                        weight = float(weight_item.text()) / 100.0
                        if weight > 0:
                            holdings[ticker] = weight
                    except ValueError:
                        pass
        return holdings

    def _get_tickers(self) -> List[str]:
        """Get list of tickers from table."""
        tickers = []
        for row in range(self.holdings_table.rowCount()):
            ticker_item = self.holdings_table.item(row, 0)
            if ticker_item:
                ticker = ticker_item.text().strip().upper()
                if ticker:
                    tickers.append(ticker)
        return tickers

    def _validate_and_fetch(self):
        """Validate tickers and fetch data."""
        tickers = self._get_tickers()
        if not tickers:
            QMessageBox.warning(self, "No Tickers", "Please enter at least one ticker.")
            return

        # Start fetching
        self.btn_validate.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(tickers))
        self.progress_bar.setValue(0)
        self.progress_label.setText("Fetching ticker data...")

        self._fetch_worker = TickerFetchWorker(tickers)
        self._fetch_worker.progress.connect(self._on_fetch_progress)
        self._fetch_worker.finished.connect(self._on_fetch_finished)
        self._fetch_worker.error.connect(self._on_fetch_error)
        self._fetch_worker.start()

    def _on_fetch_progress(self, message: str, current: int, total: int):
        """Handle fetch progress updates."""
        self.progress_bar.setValue(current)
        self.progress_label.setText(message)

    def _on_fetch_finished(self, results: dict):
        """Handle fetch completion."""
        self.btn_validate.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")

        self.ticker_data = results

        # Update table with fetched data
        has_errors = False
        for row in range(self.holdings_table.rowCount()):
            ticker_item = self.holdings_table.item(row, 0)
            if not ticker_item:
                continue

            ticker = ticker_item.text().strip().upper()
            info = results.get(ticker)

            if info and info.is_valid:
                # Name
                name_item = self.holdings_table.item(row, 2)
                name_item.setText(info.name[:30] if info.name else "—")

                # Expense ratio
                expense_item = self.holdings_table.item(row, 3)
                expense_item.setText(info.expense_ratio_pct())

                # Yield
                yield_item = self.holdings_table.item(row, 4)
                yield_item.setText(info.dividend_yield_pct())

                # Clear any error highlighting
                for col in range(5):
                    item = self.holdings_table.item(row, col)
                    if item:
                        item.setBackground(QColor(255, 255, 255))
            else:
                # Highlight invalid row
                has_errors = True
                for col in range(5):
                    item = self.holdings_table.item(row, col)
                    if item:
                        item.setBackground(QColor(255, 200, 200))

                # Show error in name column
                name_item = self.holdings_table.item(row, 2)
                error_msg = info.error_message if info else "Unknown error"
                name_item.setText(f"Error: {error_msg[:20]}...")

        if has_errors:
            QMessageBox.warning(
                self, "Validation Errors",
                "Some tickers could not be validated. Check highlighted rows."
            )

        # Update portfolio stats
        self._update_portfolio_stats()

    def _on_fetch_error(self, error: str):
        """Handle fetch error."""
        self.btn_validate.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        QMessageBox.critical(self, "Fetch Error", f"Error fetching data:\n{error}")

    def _update_portfolio_stats(self):
        """Update portfolio statistics."""
        holdings = self._get_holdings()
        if not holdings or not self.ticker_data:
            self._clear_stats()
            return

        from ...core.ticker_data import calculate_portfolio_stats
        stats = calculate_portfolio_stats(holdings, self.ticker_data)

        # Update labels
        if stats["weighted_expense_ratio"] is not None:
            self.lbl_expense.setText(f"{stats['weighted_expense_ratio'] * 100:.3f}%")
        else:
            self.lbl_expense.setText("N/A")

        if stats["weighted_yield"] is not None:
            self.lbl_yield.setText(f"{stats['weighted_yield'] * 100:.2f}%")
        else:
            self.lbl_yield.setText("N/A")

        if stats["weighted_return_1yr"] is not None:
            self.lbl_return_1yr.setText(f"{stats['weighted_return_1yr'] * 100:.2f}%")
        else:
            self.lbl_return_1yr.setText("N/A")

        if stats["weighted_return_5yr"] is not None:
            self.lbl_return_5yr.setText(f"{stats['weighted_return_5yr'] * 100:.2f}%")
        else:
            self.lbl_return_5yr.setText("N/A")

        if stats["weighted_return_10yr"] is not None:
            self.lbl_return_10yr.setText(f"{stats['weighted_return_10yr'] * 100:.2f}%")
        else:
            self.lbl_return_10yr.setText("N/A")

    def _clear_stats(self):
        """Clear portfolio statistics."""
        self.lbl_expense.setText("—")
        self.lbl_yield.setText("—")
        self.lbl_return_1yr.setText("—")
        self.lbl_return_5yr.setText("—")
        self.lbl_return_10yr.setText("—")

    def _send_to_hammer(self):
        """Send current portfolio to HAMMER tab."""
        holdings = self._get_holdings()
        if not holdings:
            QMessageBox.warning(self, "No Portfolio", "Please create a portfolio first.")
            return

        tickers = list(holdings.keys())
        self.hammer_view.set_portfolio(tickers, holdings)

        # Switch to HAMMER tab
        self.tabs.setCurrentIndex(1)

        QMessageBox.information(
            self, "Sent to HAMMER",
            f"Portfolio with {len(holdings)} holdings sent to HAMMER Analysis tab."
        )

    def _send_to_monte_carlo(self):
        """Send current portfolio to Monte Carlo."""
        holdings = self._get_holdings()
        if not holdings:
            QMessageBox.warning(self, "No Portfolio", "Please create a portfolio first.")
            return

        # Calculate expected return/risk from CMAs if available
        tickers = list(holdings.keys())
        weights = list(holdings.values())

        # Try to match tickers to CMA
        cma_tickers = self.cma.tickers if self.cma else []
        matched_indices = []
        for t in tickers:
            if t in cma_tickers:
                matched_indices.append(cma_tickers.index(t))

        exp_return = 0.0
        risk = 0.0

        if len(matched_indices) == len(tickers) and self.cma:
            w = np.array(weights)
            mu = np.array([self.cma.exp_returns[i] for i in matched_indices])
            cov_matrix = self.cma.cov[np.ix_(matched_indices, matched_indices)]
            exp_return = float(mu @ w)
            risk = float(np.sqrt(w @ cov_matrix @ w))

        payload = {
            "tickers": tickers,
            "weights": weights,
            "expected_return": exp_return,
            "risk": risk,
        }

        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, "selected_portfolio.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        QMessageBox.information(self, "Sent", f"Saved portfolio to {path}")

    def _compute_frontier(self):
        """Compute and plot efficient frontier."""
        if not self.cma:
            return

        try:
            self.frontier_result = build_frontier(
                self.cma.exp_returns.copy(),
                self.cma.cov.copy(),
                rf=self.cma.rf,
                k=40,
                w_max=0.50
            )
            self._plot_frontier()
        except Exception as e:
            print(f"Error computing frontier: {e}")

    def _plot_frontier(self):
        """Plot the efficient frontier."""
        if not self.frontier_result:
            return

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Plot frontier
        ax.plot(
            self.frontier_result.risks,
            self.frontier_result.returns,
            "o-", alpha=0.7, markersize=4, label="Efficient Frontier"
        )

        # Mark max Sharpe
        ax.scatter(
            [self.frontier_result.max_sharpe_risk],
            [self.frontier_result.max_sharpe_ret],
            s=80, marker="*", color="gold", edgecolor="black",
            label=f"Max Sharpe ({self.frontier_result.max_sharpe_sr:.2f})", zorder=5
        )

        ax.set_xlabel("Risk (Std Dev)")
        ax.set_ylabel("Expected Return")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        self.canvas.draw_idle()

    def reload_models(self):
        """Reload models from file."""
        self.models = load_models()
        self._populate_model_dropdown()
