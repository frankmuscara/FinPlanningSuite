# finplan_suite/ui/views/model_manager_view.py
"""Model Manager view for editing portfolio model presets."""

import json
import os
from typing import Dict, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QGroupBox, QMessageBox, QListWidget, QListWidgetItem,
    QSplitter, QLineEdit, QTextEdit, QHeaderView, QInputDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor


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


class ModelManagerView(QWidget):
    """Model Manager for editing portfolio model presets."""

    models_changed = pyqtSignal()  # Emitted when models are saved

    def __init__(self):
        super().__init__()
        self.setObjectName("ModelManagerView")

        # Data
        self.models = load_models()
        self.current_model_name: Optional[str] = None
        self.has_unsaved_changes = False

        # UI setup
        self._setup_ui()
        self._populate_model_list()

    def _setup_ui(self):
        """Set up the UI."""
        main_layout = QVBoxLayout(self)

        # Header
        header = QLabel("Model Manager")
        header.setStyleSheet("font-size: 20px; font-weight: 600;")
        main_layout.addWidget(header)

        description = QLabel("Edit preset portfolio models. Changes are saved to data/models.json.")
        description.setStyleSheet("color: #666; margin-bottom: 10px;")
        main_layout.addWidget(description)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Model list
        left_panel = self._create_model_list_panel()
        splitter.addWidget(left_panel)

        # Right panel: Model editor
        right_panel = self._create_model_editor_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions (30% list, 70% editor)
        splitter.setSizes([300, 700])

        main_layout.addWidget(splitter, 1)

    def _create_model_list_panel(self) -> QWidget:
        """Create the model list panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 10, 0)

        # List group
        list_group = QGroupBox("Portfolio Models")
        list_layout = QVBoxLayout(list_group)

        self.model_list = QListWidget()
        self.model_list.currentItemChanged.connect(self._on_model_selected)
        list_layout.addWidget(self.model_list)

        # List buttons
        btn_row = QHBoxLayout()

        self.btn_add_model = QPushButton("Add New")
        self.btn_add_model.clicked.connect(self._add_new_model)
        btn_row.addWidget(self.btn_add_model)

        self.btn_duplicate = QPushButton("Duplicate")
        self.btn_duplicate.clicked.connect(self._duplicate_model)
        btn_row.addWidget(self.btn_duplicate)

        list_layout.addLayout(btn_row)

        layout.addWidget(list_group)

        return panel

    def _create_model_editor_panel(self) -> QWidget:
        """Create the model editor panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 0, 0, 0)

        # Editor group
        editor_group = QGroupBox("Model Editor")
        editor_layout = QVBoxLayout(editor_group)

        # Model name
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Model Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter model name...")
        self.name_edit.textChanged.connect(self._mark_unsaved)
        name_row.addWidget(self.name_edit, 1)
        editor_layout.addLayout(name_row)

        # Description
        desc_row = QHBoxLayout()
        desc_row.addWidget(QLabel("Description:"))
        self.desc_edit = QLineEdit()
        self.desc_edit.setPlaceholderText("Brief description of the model...")
        self.desc_edit.textChanged.connect(self._mark_unsaved)
        desc_row.addWidget(self.desc_edit, 1)
        editor_layout.addLayout(desc_row)

        # Holdings table
        holdings_label = QLabel("Holdings:")
        holdings_label.setStyleSheet("margin-top: 10px;")
        editor_layout.addWidget(holdings_label)

        self.holdings_table = QTableWidget()
        self.holdings_table.setColumnCount(2)
        self.holdings_table.setHorizontalHeaderLabels(["Ticker", "Weight %"])

        header = self.holdings_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

        self.holdings_table.setAlternatingRowColors(True)
        self.holdings_table.cellChanged.connect(self._on_holding_changed)
        editor_layout.addWidget(self.holdings_table)

        # Holdings buttons
        holdings_btn_row = QHBoxLayout()

        self.btn_add_holding = QPushButton("Add Holding")
        self.btn_add_holding.clicked.connect(self._add_holding_row)
        holdings_btn_row.addWidget(self.btn_add_holding)

        self.btn_remove_holding = QPushButton("Remove Holding")
        self.btn_remove_holding.clicked.connect(self._remove_holding_row)
        holdings_btn_row.addWidget(self.btn_remove_holding)

        holdings_btn_row.addStretch()

        self.weight_total_label = QLabel("Total: 0.00%")
        self.weight_total_label.setStyleSheet("font-weight: bold;")
        holdings_btn_row.addWidget(self.weight_total_label)

        editor_layout.addLayout(holdings_btn_row)

        layout.addWidget(editor_group, 1)

        # Action buttons
        action_row = QHBoxLayout()

        self.btn_save = QPushButton("Save Model")
        self.btn_save.setStyleSheet("padding: 8px 20px; font-weight: bold; background-color: #4CAF50; color: white;")
        self.btn_save.clicked.connect(self._save_current_model)
        action_row.addWidget(self.btn_save)

        self.btn_delete = QPushButton("Delete Model")
        self.btn_delete.setStyleSheet("padding: 8px 20px; background-color: #f44336; color: white;")
        self.btn_delete.clicked.connect(self._delete_current_model)
        action_row.addWidget(self.btn_delete)

        self.btn_revert = QPushButton("Revert Changes")
        self.btn_revert.clicked.connect(self._revert_changes)
        action_row.addWidget(self.btn_revert)

        action_row.addStretch()

        self.unsaved_label = QLabel("")
        self.unsaved_label.setStyleSheet("color: #f90;")
        action_row.addWidget(self.unsaved_label)

        layout.addLayout(action_row)

        # Initially disable editor
        self._set_editor_enabled(False)

        return panel

    def _populate_model_list(self):
        """Populate the model list widget."""
        self.model_list.blockSignals(True)
        self.model_list.clear()

        for name in sorted(self.models.keys()):
            item = QListWidgetItem(name)
            self.model_list.addItem(item)

        self.model_list.blockSignals(False)

    def _on_model_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Handle model selection from list."""
        if self.has_unsaved_changes and previous:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                f"You have unsaved changes to '{self.current_model_name}'. Discard?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                # Revert selection
                self.model_list.blockSignals(True)
                self.model_list.setCurrentItem(previous)
                self.model_list.blockSignals(False)
                return

        if current is None:
            self._set_editor_enabled(False)
            self.current_model_name = None
            return

        model_name = current.text()
        self._load_model_into_editor(model_name)

    def _load_model_into_editor(self, model_name: str):
        """Load a model into the editor."""
        self.current_model_name = model_name
        model = self.models.get(model_name, {})

        # Block signals during load
        self.name_edit.blockSignals(True)
        self.desc_edit.blockSignals(True)
        self.holdings_table.blockSignals(True)

        self.name_edit.setText(model_name)
        self.desc_edit.setText(model.get("description", ""))

        # Load holdings
        self.holdings_table.setRowCount(0)
        holdings = model.get("holdings", {})
        for ticker, weight in sorted(holdings.items(), key=lambda x: -x[1]):
            row = self.holdings_table.rowCount()
            self.holdings_table.insertRow(row)

            ticker_item = QTableWidgetItem(ticker)
            self.holdings_table.setItem(row, 0, ticker_item)

            weight_item = QTableWidgetItem(f"{weight * 100:.2f}")
            weight_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.holdings_table.setItem(row, 1, weight_item)

        # Unblock signals
        self.name_edit.blockSignals(False)
        self.desc_edit.blockSignals(False)
        self.holdings_table.blockSignals(False)

        self._update_weight_total()
        self._set_editor_enabled(True)
        self.has_unsaved_changes = False
        self.unsaved_label.setText("")

    def _set_editor_enabled(self, enabled: bool):
        """Enable or disable the editor controls."""
        self.name_edit.setEnabled(enabled)
        self.desc_edit.setEnabled(enabled)
        self.holdings_table.setEnabled(enabled)
        self.btn_add_holding.setEnabled(enabled)
        self.btn_remove_holding.setEnabled(enabled)
        self.btn_save.setEnabled(enabled)
        self.btn_delete.setEnabled(enabled)
        self.btn_revert.setEnabled(enabled)

        if not enabled:
            self.name_edit.clear()
            self.desc_edit.clear()
            self.holdings_table.setRowCount(0)
            self.weight_total_label.setText("Total: 0.00%")

    def _add_holding_row(self):
        """Add an empty holding row."""
        row = self.holdings_table.rowCount()
        self.holdings_table.insertRow(row)

        self.holdings_table.setItem(row, 0, QTableWidgetItem(""))

        weight_item = QTableWidgetItem("0.00")
        weight_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.holdings_table.setItem(row, 1, weight_item)

        self._mark_unsaved()

    def _remove_holding_row(self):
        """Remove the selected holding row."""
        current_row = self.holdings_table.currentRow()
        if current_row >= 0:
            self.holdings_table.removeRow(current_row)
            self._update_weight_total()
            self._mark_unsaved()

    def _on_holding_changed(self, row: int, col: int):
        """Handle holding cell change."""
        self._update_weight_total()
        self._mark_unsaved()

    def _update_weight_total(self):
        """Update the weight total display."""
        total = 0.0
        for row in range(self.holdings_table.rowCount()):
            item = self.holdings_table.item(row, 1)
            if item:
                try:
                    total += float(item.text())
                except ValueError:
                    pass

        self.weight_total_label.setText(f"Total: {total:.2f}%")

        if abs(total - 100.0) > 0.01:
            self.weight_total_label.setStyleSheet("font-weight: bold; color: #c00;")
        else:
            self.weight_total_label.setStyleSheet("font-weight: bold; color: #090;")

    def _mark_unsaved(self):
        """Mark that there are unsaved changes."""
        self.has_unsaved_changes = True
        self.unsaved_label.setText("Unsaved changes")

    def _get_holdings_from_table(self) -> Dict[str, float]:
        """Extract holdings dict from table."""
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

    def _save_current_model(self):
        """Save the current model."""
        new_name = self.name_edit.text().strip()
        if not new_name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a model name.")
            return

        holdings = self._get_holdings_from_table()
        if not holdings:
            QMessageBox.warning(self, "No Holdings", "Please add at least one holding.")
            return

        # Check weight total
        total = sum(holdings.values())
        if abs(total - 1.0) > 0.01:
            reply = QMessageBox.question(
                self, "Weight Total",
                f"Holdings sum to {total * 100:.2f}% instead of 100%. Save anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # If renamed, remove old entry
        if self.current_model_name and new_name != self.current_model_name:
            if new_name in self.models:
                reply = QMessageBox.question(
                    self, "Model Exists",
                    f"A model named '{new_name}' already exists. Overwrite?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return

            del self.models[self.current_model_name]

        # Save model
        self.models[new_name] = {
            "description": self.desc_edit.text().strip(),
            "holdings": holdings,
        }

        # Persist to file
        save_models(self.models)

        # Update UI
        self.current_model_name = new_name
        self.has_unsaved_changes = False
        self.unsaved_label.setText("")

        self._populate_model_list()

        # Select the saved model
        for i in range(self.model_list.count()):
            if self.model_list.item(i).text() == new_name:
                self.model_list.setCurrentRow(i)
                break

        # Emit signal for other views
        self.models_changed.emit()

        QMessageBox.information(self, "Saved", f"Model '{new_name}' saved successfully.")

    def _delete_current_model(self):
        """Delete the current model."""
        if not self.current_model_name:
            return

        reply = QMessageBox.question(
            self, "Delete Model",
            f"Are you sure you want to delete '{self.current_model_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            del self.models[self.current_model_name]
            save_models(self.models)

            self.current_model_name = None
            self.has_unsaved_changes = False
            self._set_editor_enabled(False)
            self._populate_model_list()
            self.models_changed.emit()

            QMessageBox.information(self, "Deleted", "Model deleted successfully.")

    def _revert_changes(self):
        """Revert unsaved changes."""
        if self.current_model_name:
            self._load_model_into_editor(self.current_model_name)

    def _add_new_model(self):
        """Add a new model."""
        if self.has_unsaved_changes:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Discard?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        name, ok = QInputDialog.getText(
            self, "New Model",
            "Enter name for new model:",
            QLineEdit.EchoMode.Normal
        )

        if ok and name.strip():
            name = name.strip()
            if name in self.models:
                QMessageBox.warning(self, "Model Exists", f"A model named '{name}' already exists.")
                return

            # Create empty model
            self.models[name] = {
                "description": "",
                "holdings": {},
            }

            self._populate_model_list()

            # Select the new model
            for i in range(self.model_list.count()):
                if self.model_list.item(i).text() == name:
                    self.model_list.setCurrentRow(i)
                    break

            # Add an empty row
            self._add_holding_row()

    def _duplicate_model(self):
        """Duplicate the current model."""
        if not self.current_model_name:
            QMessageBox.warning(self, "No Selection", "Please select a model to duplicate.")
            return

        name, ok = QInputDialog.getText(
            self, "Duplicate Model",
            "Enter name for duplicate:",
            QLineEdit.EchoMode.Normal,
            f"{self.current_model_name} Copy"
        )

        if ok and name.strip():
            name = name.strip()
            if name in self.models:
                QMessageBox.warning(self, "Model Exists", f"A model named '{name}' already exists.")
                return

            # Copy model
            original = self.models[self.current_model_name]
            self.models[name] = {
                "description": original.get("description", ""),
                "holdings": dict(original.get("holdings", {})),
            }

            save_models(self.models)
            self._populate_model_list()

            # Select the new model
            for i in range(self.model_list.count()):
                if self.model_list.item(i).text() == name:
                    self.model_list.setCurrentRow(i)
                    break

            self.models_changed.emit()

    def refresh_models(self):
        """Reload models from file."""
        self.models = load_models()
        self._populate_model_list()
        self._set_editor_enabled(False)
        self.current_model_name = None
        self.has_unsaved_changes = False
