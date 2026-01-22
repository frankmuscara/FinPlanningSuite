"""Strategy configuration widget for HAMMER."""

from datetime import date
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QDateEdit,
)
from PyQt6.QtCore import pyqtSignal, QDate


class StrategyConfigWidget(QWidget):
    """Widget for configuring HAMMER strategy parameters."""

    configChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Strategy Mode Selection
        mode_group = QGroupBox("Rebalancing Strategy")
        mode_layout = QVBoxLayout(mode_group)

        self.mode_buttons = QButtonGroup(self)
        self.mode_buttons.setExclusive(True)

        strategies = [
            ("hammer", "HAMMER (Drift + VIX Gate)", "Recommended - blocks intra-equity rebalancing during market stress"),
            ("shield", "SHIELD (Periodic + VIX Gate)", "Quarterly rebalancing blocked during VIX inversion"),
            ("drift", "Drift-Based", "Rebalance when allocation drifts beyond threshold"),
            ("periodic", "Periodic", "Rebalance on fixed schedule"),
            ("buy_hold", "Buy & Hold", "No rebalancing after initial investment"),
        ]

        for i, (mode, label, desc) in enumerate(strategies):
            row = QHBoxLayout()
            rb = QRadioButton(label)
            rb.setProperty("mode", mode)
            if mode == "hammer":
                rb.setChecked(True)
            self.mode_buttons.addButton(rb, i)
            row.addWidget(rb)

            desc_label = QLabel(f"<i style='color: gray;'>{desc}</i>")
            desc_label.setWordWrap(True)
            row.addWidget(desc_label, 1)

            mode_layout.addLayout(row)

        self.mode_buttons.buttonClicked.connect(self._on_mode_changed)
        layout.addWidget(mode_group)

        # Parameters section
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)

        # Drift threshold
        drift_row = QHBoxLayout()
        drift_row.addWidget(QLabel("Drift Threshold:"))
        self.drift_spin = QDoubleSpinBox()
        self.drift_spin.setRange(1.0, 50.0)
        self.drift_spin.setSingleStep(1.0)
        self.drift_spin.setValue(5.0)
        self.drift_spin.setSuffix("%")
        self.drift_spin.setDecimals(0)
        # Display as percentage (5 = 5%, internally converted to 0.05)
        self.drift_spin.valueChanged.connect(self.configChanged.emit)
        drift_row.addWidget(self.drift_spin)
        drift_row.addWidget(QLabel("(rebalance when any asset drifts this far from target)"))
        drift_row.addStretch()
        params_layout.addLayout(drift_row)

        # Periodic frequency (shown only for periodic mode)
        freq_row = QHBoxLayout()
        freq_row.addWidget(QLabel("Frequency:"))
        self.freq_combo = QComboBox()
        self.freq_combo.addItems(["Monthly", "Quarterly", "Annual"])
        self.freq_combo.setCurrentIndex(1)  # Quarterly default
        self.freq_combo.setEnabled(False)  # Disabled by default (not periodic mode)
        self.freq_combo.currentIndexChanged.connect(self.configChanged.emit)
        freq_row.addWidget(self.freq_combo)
        freq_row.addStretch()
        params_layout.addLayout(freq_row)

        layout.addWidget(params_group)

        # Backtest Period
        period_group = QGroupBox("Backtest Period")
        period_layout = QHBoxLayout(period_group)

        period_layout.addWidget(QLabel("Start:"))
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate(date.today().year - 10, 1, 1))
        self.start_date.dateChanged.connect(self.configChanged.emit)
        period_layout.addWidget(self.start_date)

        period_layout.addWidget(QLabel("End:"))
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        self.end_date.dateChanged.connect(self.configChanged.emit)
        period_layout.addWidget(self.end_date)

        period_layout.addWidget(QLabel("Capital:"))
        self.capital_spin = QSpinBox()
        self.capital_spin.setRange(1000, 100000000)
        self.capital_spin.setSingleStep(10000)
        self.capital_spin.setValue(100000)
        self.capital_spin.setPrefix("$")
        self.capital_spin.valueChanged.connect(self.configChanged.emit)
        period_layout.addWidget(self.capital_spin)

        period_layout.addStretch()
        layout.addWidget(period_group)

        layout.addStretch()

    def _on_mode_changed(self, button):
        """Handle strategy mode change."""
        mode = button.property("mode")

        # Enable/disable drift threshold based on mode
        self.drift_spin.setEnabled(mode in ("drift", "hammer"))

        # Enable/disable frequency based on mode (periodic and shield use frequency)
        self.freq_combo.setEnabled(mode in ("periodic", "shield"))

        self.configChanged.emit()

    def get_mode(self) -> str:
        """Get selected strategy mode."""
        checked = self.mode_buttons.checkedButton()
        return checked.property("mode") if checked else "hammer"

    def get_drift_threshold(self) -> float:
        """Get drift threshold as decimal (0.05 = 5%)."""
        return self.drift_spin.value() / 100.0

    def get_frequency(self) -> str:
        """Get rebalance frequency."""
        return self.freq_combo.currentText().lower()

    def get_start_date(self) -> date:
        """Get backtest start date."""
        qdate = self.start_date.date()
        return date(qdate.year(), qdate.month(), qdate.day())

    def get_end_date(self) -> date:
        """Get backtest end date."""
        qdate = self.end_date.date()
        return date(qdate.year(), qdate.month(), qdate.day())

    def get_initial_capital(self) -> float:
        """Get initial capital."""
        return float(self.capital_spin.value())

    def get_config(self) -> dict:
        """Get full configuration as dictionary."""
        return {
            "mode": self.get_mode(),
            "drift_threshold": self.get_drift_threshold(),
            "frequency": self.get_frequency(),
            "start_date": self.get_start_date(),
            "end_date": self.get_end_date(),
            "initial_capital": self.get_initial_capital(),
        }
