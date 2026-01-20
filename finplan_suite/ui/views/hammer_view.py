"""HAMMER Analysis view for backtesting with VIX-gated rebalancing."""

import os
import json
from datetime import date
from typing import Optional, Dict

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMessageBox,
    QSplitter,
    QGroupBox,
    QComboBox,
    QProgressBar,
    QTextEdit,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from finplan_suite.ui.widgets.strategy_config import StrategyConfigWidget
from finplan_suite.ui.widgets.metrics_table import MetricsTableWidget


class BacktestWorker(QThread):
    """Worker thread for running backtests without blocking UI."""

    finished = pyqtSignal(object, object)  # hammer_result, drift_result
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, portfolio_config, drift_threshold):
        super().__init__()
        self.portfolio_config = portfolio_config
        self.drift_threshold = drift_threshold

    def run(self):
        try:
            self.progress.emit("Fetching price data...")

            from finplan_suite.core.hammer_bridge import run_comparison_backtest
            hammer_result, drift_result = run_comparison_backtest(
                self.portfolio_config,
                self.drift_threshold,
            )

            self.finished.emit(hammer_result, drift_result)
        except Exception as e:
            self.error.emit(str(e))


class HammerView(QWidget):
    """HAMMER Analysis tab for Portfolio Builder."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HammerView")

        self._current_weights = None
        self._current_tickers = None
        self._hammer_result = None
        self._drift_result = None
        self._worker = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("HAMMER Analysis — VIX-Gated Rebalancing Backtest")
        header.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(header)

        desc = QLabel(
            "HAMMER blocks intra-equity rebalancing during VIX curve inversion (market panic), "
            "while still allowing asset allocation adjustments. Compare against traditional drift-based rebalancing."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #555; margin-bottom: 10px;")
        layout.addWidget(desc)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Configuration
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)

        # Portfolio info
        portfolio_group = QGroupBox("Portfolio")
        portfolio_layout = QVBoxLayout(portfolio_group)

        self.portfolio_label = QLabel("No portfolio loaded. Set weights in Efficient Frontier tab first.")
        self.portfolio_label.setWordWrap(True)
        portfolio_layout.addWidget(self.portfolio_label)

        bench_row = QHBoxLayout()
        bench_row.addWidget(QLabel("Benchmark:"))
        self.benchmark_combo = QComboBox()
        self.benchmark_combo.addItems(["SPY", "QQQ", "VTI", "IWM", "AGG"])
        bench_row.addWidget(self.benchmark_combo)
        bench_row.addStretch()
        portfolio_layout.addLayout(bench_row)

        left_layout.addWidget(portfolio_group)

        # Strategy config
        self.strategy_widget = StrategyConfigWidget()
        left_layout.addWidget(self.strategy_widget)

        # Run button
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Backtest Comparison")
        self.btn_run.setStyleSheet("padding: 10px; font-weight: bold;")
        self.btn_run.clicked.connect(self._run_backtest)
        btn_row.addWidget(self.btn_run)
        left_layout.addLayout(btn_row)

        # Progress
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #666;")
        left_layout.addWidget(self.progress_label)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # Right panel: Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 0, 0)

        # Chart
        chart_group = QGroupBox("NAV Comparison")
        chart_layout = QVBoxLayout(chart_group)

        self.fig = Figure(figsize=(8, 4), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        chart_layout.addWidget(self.canvas)

        right_layout.addWidget(chart_group, 2)

        # Metrics table
        self.metrics_table = MetricsTableWidget()
        self.metrics_table.set_title("Performance Metrics Comparison")
        right_layout.addWidget(self.metrics_table, 1)

        # Export buttons
        export_row = QHBoxLayout()
        self.btn_export_csv = QPushButton("Export Results (CSV)")
        self.btn_export_csv.clicked.connect(self._export_csv)
        self.btn_export_csv.setEnabled(False)
        export_row.addWidget(self.btn_export_csv)

        self.btn_summary = QPushButton("Generate Client Summary")
        self.btn_summary.clicked.connect(self._generate_summary)
        self.btn_summary.setEnabled(False)
        export_row.addWidget(self.btn_summary)

        export_row.addStretch()
        right_layout.addLayout(export_row)

        splitter.addWidget(right_panel)

        # Set splitter sizes (40% left, 60% right)
        splitter.setSizes([400, 600])

        layout.addWidget(splitter)

    def set_portfolio(self, tickers: list, weights: dict):
        """Set the portfolio to analyze.

        Called from PortfolioView when weights change.

        Args:
            tickers: List of ticker symbols
            weights: Dict mapping ticker to weight (0-1)
        """
        self._current_tickers = tickers
        self._current_weights = weights

        # Filter to active weights
        active = {t: w for t, w in weights.items() if w > 0.001}

        if not active:
            self.portfolio_label.setText("No portfolio loaded. Set weights in Efficient Frontier tab first.")
            self.btn_run.setEnabled(False)
            return

        # Format portfolio summary
        lines = ["<b>Current Allocation:</b>"]
        for ticker, weight in sorted(active.items(), key=lambda x: -x[1]):
            lines.append(f"  {ticker}: {weight:.1%}")

        self.portfolio_label.setText("<br>".join(lines))
        self.btn_run.setEnabled(True)

    def _run_backtest(self):
        """Run HAMMER vs Drift comparison backtest."""
        if not self._current_weights:
            QMessageBox.warning(self, "No Portfolio", "Please set portfolio weights first.")
            return

        # Get configuration
        config = self.strategy_widget.get_config()

        # Build portfolio config
        from finplan_suite.core.hammer_bridge import portfolio_weights_to_hammer_config

        try:
            portfolio_config = portfolio_weights_to_hammer_config(
                tickers=self._current_tickers,
                weights=self._current_weights,
                benchmark=self.benchmark_combo.currentText(),
                initial_capital=config["initial_capital"],
                start_date=config["start_date"],
                end_date=config["end_date"],
            )
        except Exception as e:
            QMessageBox.critical(self, "Configuration Error", str(e))
            return

        # Disable UI during backtest
        self.btn_run.setEnabled(False)
        self.progress_label.setText("Running backtest...")

        # Run in worker thread
        self._worker = BacktestWorker(portfolio_config, config["drift_threshold"])
        self._worker.finished.connect(self._on_backtest_finished)
        self._worker.error.connect(self._on_backtest_error)
        self._worker.progress.connect(lambda msg: self.progress_label.setText(msg))
        self._worker.start()

    def _on_backtest_finished(self, hammer_result, drift_result):
        """Handle backtest completion."""
        self._hammer_result = hammer_result
        self._drift_result = drift_result

        self.btn_run.setEnabled(True)
        self.btn_export_csv.setEnabled(True)
        self.btn_summary.setEnabled(True)
        self.progress_label.setText("Backtest complete!")

        # Compute metrics
        from finplan_suite.core.hammer_bridge import (
            compute_result_metrics,
            metrics_to_comparison_dict,
        )

        hammer_metrics = compute_result_metrics(hammer_result)
        drift_metrics = compute_result_metrics(drift_result)

        # Update metrics table
        comparison = metrics_to_comparison_dict(hammer_metrics, drift_metrics)
        self.metrics_table.set_comparison_data(comparison)

        # Plot results
        self._plot_results(hammer_result, drift_result)

    def _on_backtest_error(self, error_msg):
        """Handle backtest error."""
        self.btn_run.setEnabled(True)
        self.progress_label.setText("")
        QMessageBox.critical(self, "Backtest Error", f"Error running backtest:\n{error_msg}")

    def _plot_results(self, hammer_result, drift_result):
        """Plot NAV comparison chart."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Normalize to $100 start for comparison
        hammer_norm = 100 * hammer_result.nav / hammer_result.nav.iloc[0]
        drift_norm = 100 * drift_result.nav / drift_result.nav.iloc[0]
        bench_norm = 100 * hammer_result.benchmark_nav / hammer_result.benchmark_nav.iloc[0]

        # Plot NAV lines
        ax.plot(hammer_norm.index, hammer_norm.values, label="HAMMER", linewidth=2, color="#2E86AB")
        ax.plot(drift_norm.index, drift_norm.values, label="Traditional Drift", linewidth=2, color="#A23B72", alpha=0.8)
        ax.plot(bench_norm.index, bench_norm.values, label="Benchmark", linewidth=1, color="#666", linestyle="--", alpha=0.6)

        # Shade VIX backwardation regions
        if hammer_result.vix_slope is not None:
            from finplan_suite.core.hammer.vix import get_blocked_regions
            blocked = get_blocked_regions(hammer_result.vix_slope)

            for _, row in blocked.iterrows():
                ax.axvspan(row["start"], row["end"], alpha=0.15, color="red", label="_nolegend_")

            # Add one entry for legend
            if len(blocked) > 0:
                ax.axvspan(blocked.iloc[0]["start"], blocked.iloc[0]["start"],
                          alpha=0.15, color="red", label="VIX Inverted")

        # Mark partial rebalance events
        partial_events = hammer_result.partial_events
        if partial_events:
            partial_dates = [e.date for e in partial_events]
            partial_values = [hammer_norm.loc[str(d)] if str(d) in hammer_norm.index else None
                            for d in partial_dates]
            partial_values = [v for v in partial_values if v is not None]
            if partial_values:
                ax.scatter([partial_dates[i] for i, v in enumerate(partial_values) if v],
                          partial_values, marker="^", s=50, color="orange",
                          label=f"Equity Frozen ({len(partial_events)})", zorder=5)

        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value (Normalized to $100)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Format final values annotation
        final_hammer = hammer_norm.iloc[-1]
        final_drift = drift_norm.iloc[-1]
        ax.annotate(f"${final_hammer:.0f}", xy=(hammer_norm.index[-1], final_hammer),
                   xytext=(5, 0), textcoords="offset points", fontsize=9, color="#2E86AB")
        ax.annotate(f"${final_drift:.0f}", xy=(drift_norm.index[-1], final_drift),
                   xytext=(5, 0), textcoords="offset points", fontsize=9, color="#A23B72")

        self.canvas.draw_idle()

    def _export_csv(self):
        """Export results to CSV."""
        if not self._hammer_result:
            return

        from finplan_suite.core.hammer_bridge import save_backtest_result

        try:
            path = save_backtest_result(self._hammer_result, "hammer_backtest")
            QMessageBox.information(self, "Exported", f"Results saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _generate_summary(self):
        """Generate client-facing summary."""
        if not self._hammer_result or not self._drift_result:
            return

        from finplan_suite.core.hammer_bridge import (
            compute_result_metrics,
            generate_client_summary,
        )

        hammer_metrics = compute_result_metrics(self._hammer_result)
        drift_metrics = compute_result_metrics(self._drift_result)

        summary = generate_client_summary(
            self._hammer_result,
            self._drift_result,
            hammer_metrics,
            drift_metrics,
        )

        # Show in dialog
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Client Summary")
        dialog.setText("Summary generated. Copy the text below:")
        dialog.setDetailedText(summary)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.exec()
