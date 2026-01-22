"""Metrics comparison table widget for HAMMER."""

from typing import Dict, Optional
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLabel,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont


class MetricsTableWidget(QWidget):
    """Widget displaying performance metrics comparison."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel("Performance Metrics")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.title_label)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Metric", "Strategy", "Benchmark", "Difference"])

        # Style header
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        layout.addWidget(self.table)

    def set_title(self, title: str):
        """Set the table title."""
        self.title_label.setText(title)

    def set_comparison_data(self, comparison: Dict[str, Dict[str, str]]):
        """Populate table with comparison data.

        Args:
            comparison: Dict mapping metric name to {HAMMER, Benchmark, Difference}
        """
        self.table.setRowCount(len(comparison))

        for row, (metric_name, values) in enumerate(comparison.items()):
            # Metric name
            name_item = QTableWidgetItem(metric_name)
            name_item.setFont(QFont("", -1, QFont.Weight.Bold))
            self.table.setItem(row, 0, name_item)

            # Strategy value
            strategy_item = QTableWidgetItem(values.get("Strategy", "N/A"))
            strategy_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 1, strategy_item)

            # Benchmark value
            benchmark_item = QTableWidgetItem(values.get("Benchmark", "N/A"))
            benchmark_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 2, benchmark_item)

            # Difference
            diff_str = values.get("Difference", "N/A")
            diff_item = QTableWidgetItem(diff_str)
            diff_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            # Color code the difference
            if "✓" in diff_str:
                diff_item.setForeground(QColor(0, 128, 0))  # Green for positive
                diff_item.setFont(QFont("", -1, QFont.Weight.Bold))
            elif diff_str.startswith("-") and "N/A" not in diff_str and "—" not in diff_str:
                # Check if this is a "lower is better" metric
                if metric_name in ("Max Drawdown", "Volatility", "Total Turnover"):
                    diff_item.setForeground(QColor(0, 128, 0))  # Green
                else:
                    diff_item.setForeground(QColor(180, 0, 0))  # Red

            self.table.setItem(row, 3, diff_item)

        self.table.resizeRowsToContents()

    def set_single_strategy_data(self, metrics_dict: Dict[str, str], strategy_name: str = "Results"):
        """Populate table with single strategy metrics.

        Args:
            metrics_dict: Dict mapping metric name to formatted value
            strategy_name: Name for the strategy column
        """
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Metric", strategy_name])

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

        self.table.setRowCount(len(metrics_dict))

        for row, (metric_name, value) in enumerate(metrics_dict.items()):
            name_item = QTableWidgetItem(metric_name)
            name_item.setFont(QFont("", -1, QFont.Weight.Bold))
            self.table.setItem(row, 0, name_item)

            value_item = QTableWidgetItem(value)
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 1, value_item)

        self.table.resizeRowsToContents()

    def clear(self):
        """Clear the table."""
        self.table.setRowCount(0)
