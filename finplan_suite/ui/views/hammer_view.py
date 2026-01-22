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
import pandas as pd

from finplan_suite.ui.widgets.strategy_config import StrategyConfigWidget
from finplan_suite.core.hammer.vix import get_blocked_regions
from finplan_suite.ui.widgets.metrics_table import MetricsTableWidget


class BacktestWorker(QThread):
    """Worker thread for running backtests without blocking UI."""

    finished = pyqtSignal(object, object)  # strategy_result, benchmark_metrics
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, portfolio_config, strategy_config):
        super().__init__()
        self.portfolio_config = portfolio_config
        self.strategy_config = strategy_config

    def run(self):
        try:
            self.progress.emit("Fetching price data...")

            from finplan_suite.core.hammer_bridge import run_comparison_backtest
            strategy_result, benchmark_metrics = run_comparison_backtest(
                self.portfolio_config,
                self.strategy_config,
            )

            self.finished.emit(strategy_result, benchmark_metrics)
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
        self._benchmark_metrics = None
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
            "while still allowing asset allocation adjustments. Compare your portfolio against a benchmark."
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
        # Organized by category with clear labels
        benchmarks = [
            # Broad Market Equity
            ("SPY", "SPY - S&P 500"),
            ("QQQ", "QQQ - Nasdaq 100"),
            ("VTI", "VTI - Total US Stock"),
            ("IWM", "IWM - Russell 2000"),
            ("VEA", "VEA - Developed Intl"),
            ("VWO", "VWO - Emerging Markets"),
            # Fixed Income
            ("AGG", "AGG - US Aggregate Bond"),
            ("BND", "BND - Total Bond Market"),
            ("TLT", "TLT - 20+ Year Treasury"),
            # Vanguard LifeStrategy Funds (Asset Allocation)
            ("VASGX", "VASGX - LifeStrategy Growth (80/20)"),
            ("VSMGX", "VSMGX - LifeStrategy Mod Growth (60/40)"),
            ("VSCGX", "VSCGX - LifeStrategy Conserv (40/60)"),
            ("VASIX", "VASIX - LifeStrategy Income (20/80)"),
            # Other Balanced Funds
            ("AOR", "AOR - iShares Growth Alloc (60/40)"),
            ("AOM", "AOM - iShares Moderate Alloc (40/60)"),
            ("AOK", "AOK - iShares Conserv Alloc (30/70)"),
            ("AOA", "AOA - iShares Aggress Alloc (80/20)"),
        ]
        for ticker, label in benchmarks:
            self.benchmark_combo.addItem(label, ticker)
        self.benchmark_combo.setCurrentIndex(0)  # Default to SPY
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
        """Run strategy backtest and compare vs benchmark."""
        if not self._current_weights:
            QMessageBox.warning(self, "No Portfolio", "Please set portfolio weights first.")
            return

        # Get configuration from UI
        config = self.strategy_widget.get_config()

        # Build portfolio config
        from finplan_suite.core.hammer_bridge import (
            portfolio_weights_to_hammer_config,
            create_strategy_config,
        )

        try:
            portfolio_config = portfolio_weights_to_hammer_config(
                tickers=self._current_tickers,
                weights=self._current_weights,
                benchmark=self.benchmark_combo.currentData(),
                initial_capital=config["initial_capital"],
                start_date=config["start_date"],
                end_date=config["end_date"],
            )

            # Create strategy config from UI settings
            strategy_config = create_strategy_config(
                mode=config["mode"],
                drift_threshold=config["drift_threshold"],
                frequency=config["frequency"],
            )
        except Exception as e:
            QMessageBox.critical(self, "Configuration Error", str(e))
            return

        # Disable UI during backtest
        self.btn_run.setEnabled(False)
        self.progress_label.setText(f"Running {config['mode'].upper()} backtest...")

        # Run in worker thread
        self._worker = BacktestWorker(portfolio_config, strategy_config)
        self._worker.finished.connect(self._on_backtest_finished)
        self._worker.error.connect(self._on_backtest_error)
        self._worker.progress.connect(lambda msg: self.progress_label.setText(msg))
        self._worker.start()

    def _on_backtest_finished(self, strategy_result, benchmark_metrics):
        """Handle backtest completion."""
        self._hammer_result = strategy_result  # Keep name for compatibility
        self._benchmark_metrics = benchmark_metrics

        self.btn_run.setEnabled(True)
        self.btn_export_csv.setEnabled(True)
        self.btn_summary.setEnabled(True)

        # Show strategy mode in completion message
        mode = strategy_result.strategy_config.mode.value.upper()
        self.progress_label.setText(f"Backtest complete! ({mode})")

        # Compute strategy metrics
        from finplan_suite.core.hammer_bridge import (
            compute_result_metrics,
            metrics_to_comparison_dict,
        )

        strategy_metrics = compute_result_metrics(strategy_result)

        # Update metrics table (comparing Strategy vs Benchmark)
        comparison = metrics_to_comparison_dict(strategy_metrics, benchmark_metrics)
        self.metrics_table.set_comparison_data(comparison)

        # Plot results
        self._plot_results(strategy_result)

    def _on_backtest_error(self, error_msg):
        """Handle backtest error."""
        self.btn_run.setEnabled(True)
        self.progress_label.setText("")
        QMessageBox.critical(self, "Backtest Error", f"Error running backtest:\n{error_msg}")

    def _plot_results(self, strategy_result):
        """Plot NAV comparison chart."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Normalize to $100 start for comparison
        strategy_norm = 100 * strategy_result.nav / strategy_result.nav.iloc[0]
        bench_norm = 100 * strategy_result.benchmark_nav / strategy_result.benchmark_nav.iloc[0]

        # Get strategy and benchmark names
        strategy_mode = strategy_result.strategy_config.mode.value.upper()
        benchmark_name = self.benchmark_combo.currentData() or "Benchmark"

        # Plot NAV lines
        ax.plot(strategy_norm.index, strategy_norm.values, label=f"{strategy_mode} Portfolio", linewidth=2.5, color="#2E86AB")
        ax.plot(bench_norm.index, bench_norm.values, label=f"{benchmark_name} (Benchmark)", linewidth=2, color="#A23B72", alpha=0.9)

        # Shade VIX backwardation regions (only for HAMMER mode)
        if strategy_result.vix_slope is not None and strategy_mode == "HAMMER":
            blocked = get_blocked_regions(strategy_result.vix_slope)

            for _, row in blocked.iterrows():
                ax.axvspan(row["start"], row["end"], alpha=0.15, color="red", label="_nolegend_")

            # Add one entry for legend
            if len(blocked) > 0:
                ax.axvspan(blocked.iloc[0]["start"], blocked.iloc[0]["start"],
                          alpha=0.15, color="red", label="VIX Inverted (Rebal Blocked)")

        # Mark blocked and partial rebalance events
        blocked_events = strategy_result.blocked_events
        partial_events = strategy_result.partial_events

        # Show blocked events (triangles pointing down)
        if blocked_events:
            blocked_dates = [e.date for e in blocked_events]
            blocked_values = []
            for d in blocked_dates:
                try:
                    blocked_values.append(strategy_norm.loc[pd.Timestamp(d)])
                except KeyError:
                    blocked_values.append(None)
            valid_blocked = [(d, v) for d, v in zip(blocked_dates, blocked_values) if v is not None]
            if valid_blocked:
                plot_dates, plot_values = zip(*valid_blocked)
                ax.scatter(plot_dates, plot_values, marker="v", s=50, color="red",
                          label=f"Rebalance Blocked ({len(blocked_events)})", zorder=5)

        # Show partial events (triangles pointing up)
        if partial_events:
            partial_dates = [e.date for e in partial_events]
            partial_values = []
            for d in partial_dates:
                try:
                    partial_values.append(strategy_norm.loc[pd.Timestamp(d)])
                except KeyError:
                    partial_values.append(None)
            valid_points = [(d, v) for d, v in zip(partial_dates, partial_values) if v is not None]
            if valid_points:
                plot_dates, plot_values = zip(*valid_points)
                ax.scatter(plot_dates, plot_values, marker="^", s=50, color="orange",
                          label=f"Equity Frozen ({len(partial_events)})", zorder=5)

        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value (Normalized to $100)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Format final values annotation
        final_strategy = strategy_norm.iloc[-1]
        final_bench = bench_norm.iloc[-1]
        ax.annotate(f"${final_strategy:.0f}", xy=(strategy_norm.index[-1], final_strategy),
                   xytext=(5, 5), textcoords="offset points", fontsize=9, color="#2E86AB", fontweight="bold")
        ax.annotate(f"${final_bench:.0f}", xy=(bench_norm.index[-1], final_bench),
                   xytext=(5, -10), textcoords="offset points", fontsize=9, color="#A23B72", fontweight="bold")

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
        if not self._hammer_result or not self._benchmark_metrics:
            return

        from finplan_suite.core.hammer_bridge import (
            compute_result_metrics,
            generate_client_summary,
        )

        hammer_metrics = compute_result_metrics(self._hammer_result)
        benchmark_name = self.benchmark_combo.currentData() or "Benchmark"

        summary = generate_client_summary(
            self._hammer_result,
            hammer_metrics,
            self._benchmark_metrics,
            benchmark_name=benchmark_name,
        )

        # Show in dialog
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Client Summary")
        dialog.setText("Summary generated. Copy the text below:")
        dialog.setDetailedText(summary)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.exec()
