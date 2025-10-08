# finplan_suite/ui/views/economic_view.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton, QGroupBox, QGridLayout, QMessageBox
)
from PyQt6.QtCore import Qt

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ...core.forecasting import MacroInputs  # (kept for CPI/r*/TP flat baselines)
from ...core.cma import derive_cma_from_macro, save_cma_json
from ...core.forecast_models import GDPInputs, GDPGrowthForecaster, CPIInputs, CPIForecaster, RStarInputs, RStarForecaster, TPInputs, TPForecaster


class EconomicView(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("EconomicView")

        root = QVBoxLayout(self)

        title = QLabel("Economic Forecasts")
        title.setStyleSheet("font-size:20px; font-weight:600;")
        root.addWidget(title)

        desc = QLabel("Set long-run macro assumptions. Optionally fetch FRED data to model GDP growth. Save CMAs to drive Portfolio Builder.")
        desc.setWordWrap(True); desc.setStyleSheet("color:#555;")
        root.addWidget(desc)

        # ---- Inputs ----
        box = QGroupBox("Macro Inputs (long-run levels)")
        grid = QGridLayout(box)

        self.in_gdp  = QDoubleSpinBox(); self.in_gdp.setRange(-0.05, 0.06); self.in_gdp.setSingleStep(0.001); self.in_gdp.setDecimals(4); self.in_gdp.setValue(0.017)
        self.in_cpi  = QDoubleSpinBox(); self.in_cpi.setRange(-0.02, 0.08); self.in_cpi.setSingleStep(0.001); self.in_cpi.setDecimals(4); self.in_cpi.setValue(0.025)
        self.in_rsh  = QDoubleSpinBox(); self.in_rsh.setRange(-0.02, 0.05); self.in_rsh.setSingleStep(0.001); self.in_rsh.setDecimals(4); self.in_rsh.setValue(0.010)
        self.in_tp   = QDoubleSpinBox(); self.in_tp.setRange(-0.01, 0.04); self.in_tp.setSingleStep(0.001); self.in_tp.setDecimals(4); self.in_tp.setValue(0.015)

        r=0
        grid.addWidget(QLabel("Real GDP (long-run)"), r,0); grid.addWidget(self.in_gdp, r,1); r+=1
        grid.addWidget(QLabel("CPI Inflation (long-run)"), r,0); grid.addWidget(self.in_cpi, r,1); r+=1
        grid.addWidget(QLabel("Real short rate (r*)"), r,0); grid.addWidget(self.in_rsh, r,1); r+=1
        grid.addWidget(QLabel("Term premium (LT-ST)"), r,0); grid.addWidget(self.in_tp,  r,1); r+=1

        root.addWidget(box)

        # ---- Buttons ----
        row = QHBoxLayout()
        self.btn_proj = QPushButton("Project (Flat Baseline)")
        self.btn_proj.clicked.connect(self.do_project_flat)
        row.addWidget(self.btn_proj)

        self.btn_gdp_ml = QPushButton("Fetch from FRED & Recompute (GDP + CPI)")
        self.btn_gdp_ml.clicked.connect(self.do_project_with_models)
        row.addWidget(self.btn_gdp_ml)


        self.btn_push = QPushButton("Save CMA from Macro")
        self.btn_push.clicked.connect(self.push_cma)
        self.btn_push.setEnabled(False)

        row.addStretch(1)
        root.addLayout(row)

        # ---- Charts ----
        self.fig = Figure(figsize=(6,4), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas, 2)

        self.proj = None
        self.gdp_path = None  # annualized 10y path from ML
        self.cpi_path = None
        self.rstar_path = None
        self.tp_path = None

    # ---------- Flat baseline (keeps your older method for CPI/r*/TP) ----------
    def do_project_flat(self):
        # For now, just create flat paths; we’ll replace others later
        yrs = np.arange(1, 11)
        gdp = np.full_like(yrs, float(self.in_gdp.value()), dtype=float)
        cpi = np.full_like(yrs, float(self.in_cpi.value()), dtype=float)
        rsh = np.full_like(yrs, float(self.in_rsh.value()), dtype=float)
        tp  = np.full_like(yrs, float(self.in_tp.value()), dtype=float)

        self.gdp_path = gdp  # flat baseline if ML not used
        self.btn_push.setEnabled(True)
        self.plot_macro_paths(yrs, gdp, cpi, rsh, tp)

    # ---------- GDP with ML (RidgeCV over FRED drivers) ----------
    def do_project_with_models(self):
        try:
            # GDP
            gdp_model = GDPGrowthForecaster(GDPInputs(start="2010-01-01", end=None))
            gdp_res = gdp_model.run(long_run_gdp_real=float(self.in_gdp.value()))
            gdp_yearly = gdp_res.path_annualized_10y.reshape(10, 4).mean(axis=1)

            # CPI
            cpi_model = CPIForecaster(CPIInputs(start="2010-01-01", end=None))
            cpi_res = cpi_model.run(long_run_cpi=float(self.in_cpi.value()))
            cpi_yearly = cpi_res.path_yoy_10y.reshape(10, 4).mean(axis=1)

            # r* forecast
            rs_model = RStarForecaster(RStarInputs(start="2010-01-01", end=None))
            rs_res = rs_model.run(long_run_rstar=float(self.in_rsh.value()))
            rstar_yearly = rs_res.path_rstar_10y.reshape(10, 4).mean(axis=1)
            self.rstar_path = rstar_yearly

            # Term Premium
            tp_model = TPForecaster(TPInputs(start="2010-01-01", end=None))
            tp_res = tp_model.run(long_run_tp=float(self.in_tp.value()))
            tp_yearly = tp_res.path_tp_10y.reshape(10, 4).mean(axis=1)
            self.tp_path = tp_yearly

            yrs = np.arange(1, 11)
            tp  = np.full(10, float(self.in_tp.value()), dtype=float)

            self.gdp_path   = gdp_yearly
            self.cpi_path   = cpi_yearly
            self.rstar_path = rstar_yearly
            self.tp_path = tp
            self.btn_push.setEnabled(True)

            # update chart: use modeled GDP/CPI/r*, flat TP
            self.plot_macro_paths(yrs, gdp_yearly, cpi_yearly, rstar_yearly, tp_yearly)

        except Exception as e:
            QMessageBox.critical(self, "FRED/Model Error", f"Failed to fetch/train models:\n{e}")

    def plot_macro_paths(self, yrs, gdp, cpi, rsh, tp):
        self.fig.clear()
        ax1 = self.fig.add_subplot(221); ax2 = self.fig.add_subplot(222)
        ax3 = self.fig.add_subplot(223); ax4 = self.fig.add_subplot(224)

        def add(ax, y, title):
            ax.plot(yrs, y, "k", lw=2)
            ax.set_title(title); ax.grid(True, alpha=0.3); ax.set_xlim(1, yrs[-1])

        add(ax1, gdp, "Real GDP (y/y, modeled)")
        add(ax2, cpi, "CPI (y/y, modeled)")
        add(ax3, rsh, "Real Short Rate r* (modeled)")
        add(ax4, tp,  "Term Premium")

        self.canvas.draw_idle()

    def push_cma(self):
        gdp_y1 = float(self.gdp_path[0]) if self.gdp_path is not None else float(self.in_gdp.value())
        cpi_y1 = float(self.cpi_path[0]) if self.cpi_path is not None else float(self.in_cpi.value())
        rsh_y1 = float(self.rstar_path[0]) if self.rstar_path is not None else float(self.in_rsh.value())
        tp_y1  = float(self.tp_path[0])    if self.tp_path    is not None else float(self.in_tp.value())
        cma = derive_cma_from_macro(gdp_y1, cpi_y1, rsh_y1, tp_y1)
        save_cma_json(cma, path="data/cma.json")
        QMessageBox.information(self, "Saved", "CMAs saved to data/cma.json.\nPortfolio Builder will use them on next compute.")

