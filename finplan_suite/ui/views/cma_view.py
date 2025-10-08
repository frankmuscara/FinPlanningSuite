# finplan_suite/ui/views/cma_view.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QGridLayout,
    QDoubleSpinBox, QPushButton, QTableWidget, QTableWidgetItem, QMessageBox
)
from PyQt6.QtCore import Qt

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ...core.cma import (
    TICKERS, build_cma_blend, load_cma_json, save_cma_json, _default_corr, _base_vols
)

def pct_box(minv, maxv, step, val, dec=3):
    w = QDoubleSpinBox(); w.setRange(minv, maxv); w.setSingleStep(step); w.setDecimals(dec); w.setValue(val); return w

class CMAView(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("CMAView")
        self.cma = None

        root = QVBoxLayout(self)
        title = QLabel("Capital Market Assumptions")
        title.setStyleSheet("font-size:20px; font-weight:600;")
        root.addWidget(title)

        desc = QLabel("Blend macro, valuation, and prior to produce 5–10 year expected returns, covariance, and risk-free rate.")
        desc.setWordWrap(True); desc.setStyleSheet("color:#555;")
        root.addWidget(desc)

        # ---- Inputs: Macro ----
        box_macro = QGroupBox("Macro Inputs")
        g = QGridLayout(box_macro)
        self.in_gdp   = pct_box(-0.02, 0.06, 0.001, 0.017, 4)
        self.in_cpi   = pct_box(-0.01, 0.08, 0.001, 0.025, 4)
        self.in_rstar = pct_box(-0.01, 0.05, 0.001, 0.010, 4)
        self.in_tp    = pct_box(-0.01, 0.04, 0.001, 0.015, 4)

        self.in_smallprem = pct_box(0.0, 0.03, 0.001, 0.012, 4)
        self.in_creditprem= pct_box(0.0, 0.03, 0.001, 0.015, 4)

        r=0
        g.addWidget(QLabel("Real GDP (trend)"), r,0); g.addWidget(self.in_gdp, r,1); r+=1
        g.addWidget(QLabel("CPI (long-run)"),   r,0); g.addWidget(self.in_cpi, r,1); r+=1
        g.addWidget(QLabel("Real short (r*)"),  r,0); g.addWidget(self.in_rstar, r,1); r+=1
        g.addWidget(QLabel("Term premium"),     r,0); g.addWidget(self.in_tp, r,1); r+=1
        g.addWidget(QLabel("Small-cap premium"),r,0); g.addWidget(self.in_smallprem, r,1); r+=1
        g.addWidget(QLabel("Credit premium"),   r,0); g.addWidget(self.in_creditprem, r,1); r+=1

        # ---- Inputs: Valuation ----
        box_val = QGroupBox("Valuation Anchors")
        v = QGridLayout(box_val)
        self.in_ey = pct_box(0.03, 0.10, 0.001, 0.055, 4)           # large-cap earnings yield
        self.in_small_gap = pct_box(-0.02, 0.03, 0.001, 0.005, 4)   # small cheaper by +0.5%
        self.in_core_yield = pct_box(0.00, 0.10, 0.001, 0.045, 4)
        self.in_tlt_yield  = pct_box(0.00, 0.10, 0.001, 0.048, 4)
        self.in_oas        = pct_box(0.00, 0.04, 0.001, 0.015, 4)
        self.in_em_spread  = pct_box(0.00, 0.05, 0.001, 0.018, 4)
        self.in_cmd_carry  = pct_box(-0.02, 0.03, 0.001, 0.000, 4)

        r=0
        v.addWidget(QLabel("Earnings Yield (large)"), r,0); v.addWidget(self.in_ey, r,1); r+=1
        v.addWidget(QLabel("Small vs Large valuation gap"), r,0); v.addWidget(self.in_small_gap, r,1); r+=1
        v.addWidget(QLabel("Core bond yield"), r,0); v.addWidget(self.in_core_yield, r,1); r+=1
        v.addWidget(QLabel("Long Treasury yield"), r,0); v.addWidget(self.in_tlt_yield, r,1); r+=1
        v.addWidget(QLabel("Credit OAS"), r,0); v.addWidget(self.in_oas, r,1); r+=1
        v.addWidget(QLabel("EM spread"), r,0); v.addWidget(self.in_em_spread, r,1); r+=1
        v.addWidget(QLabel("Commodities carry"), r,0); v.addWidget(self.in_cmd_carry, r,1); r+=1

        # ---- Inputs: Weights ----
        box_w = QGroupBox("Blend Weights")
        w = QGridLayout(box_w)
        self.in_w_macro = pct_box(0.0, 1.0, 0.05, 0.50, 2)
        self.in_w_val   = pct_box(0.0, 1.0, 0.05, 0.30, 2)
        self.in_w_prior = pct_box(0.0, 1.0, 0.05, 0.20, 2)

        w.addWidget(QLabel("Macro weight"), 0,0); w.addWidget(self.in_w_macro, 0,1)
        w.addWidget(QLabel("Valuation weight"), 1,0); w.addWidget(self.in_w_val, 1,1)
        w.addWidget(QLabel("Prior weight"), 2,0); w.addWidget(self.in_w_prior, 2,1)

        # ---- Buttons ----
        btns = QHBoxLayout()
        self.btn_compute = QPushButton("Compute CMAs")
        self.btn_compute.clicked.connect(self.compute_cma)
        self.btn_save = QPushButton("Save CMAs")
        self.btn_save.clicked.connect(self.save_cma)
        self.btn_load = QPushButton("Load Current CMAs")
        self.btn_load.clicked.connect(self.load_cma)
        btns.addWidget(self.btn_compute); btns.addWidget(self.btn_save); btns.addWidget(self.btn_load)
        btns.addStretch(1)

        # ---- Layout grid ----
        row1 = QHBoxLayout()
        row1.addWidget(box_macro, 1)
        row1.addWidget(box_val, 1)
        row1.addWidget(box_w, 1)
        root.addLayout(row1)
        root.addLayout(btns)

        # ---- Outputs: table + heatmap ----
        self.tbl = QTableWidget(0, 4, self)
        self.tbl.setHorizontalHeaderLabels(["Ticker", "Exp Return", "Vol", "SE (ER)"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.tbl)

        self.fig = Figure(figsize=(5,3), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas, 1)

        self.status = QLabel("")
        root.addWidget(self.status)

    def compute_cma(self):
        # Gather inputs
        cma = build_cma_blend(
            # macro
            gdp=float(self.in_gdp.value()),
            cpi=float(self.in_cpi.value()),
            rstar=float(self.in_rstar.value()),
            term_premium=float(self.in_tp.value()),
            # valuation
            earnings_yield_large=float(self.in_ey.value()),
            small_vs_large_valuation_gap=float(self.in_small_gap.value()),
            core_bond_yield=float(self.in_core_yield.value()),
            tlt_yield=float(self.in_tlt_yield.value()),
            credit_oas=float(self.in_oas.value()),
            em_spread=float(self.in_em_spread.value()),
            commodities_carry=float(self.in_cmd_carry.value()),
            # weights
            w_macro=float(self.in_w_macro.value()),
            w_val=float(self.in_w_val.value()),
            w_prior=float(self.in_w_prior.value()),
            # constants
            small_premium=float(self.in_smallprem.value()),
            credit_premium=float(self.in_creditprem.value())
        )
        self.cma = cma
        self.render_outputs()
        self.status.setText(f"Computed CMAs. RF={cma.rf:.2%}")

    def render_outputs(self):
        if self.cma is None: return
        mu = self.cma.exp_returns
        vols = np.sqrt(np.diag(self.cma.cov))

        se = self.cma.meta.get("se_exp_returns", [None]*len(TICKERS))
        self.tbl.setRowCount(len(TICKERS))
        for i, t in enumerate(TICKERS):
            self.tbl.setItem(i, 0, QTableWidgetItem(t))
            self.tbl.setItem(i, 1, QTableWidgetItem(f"{mu[i]:.2%}"))
            self.tbl.setItem(i, 2, QTableWidgetItem(f"{vols[i]:.2%}"))
            self.tbl.setItem(i, 3, QTableWidgetItem("" if se[i] is None else f"{se[i]:.2%}"))

        # correlation heatmap
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        corr = self.cma.cov.copy()
        # convert to correlation for display
        s = np.sqrt(np.diag(corr))
        corr = corr / np.outer(s, s)
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(TICKERS))); ax.set_xticklabels(TICKERS, rotation=45, ha="right")
        ax.set_yticks(range(len(TICKERS))); ax.set_yticklabels(TICKERS)
        ax.set_title("Correlation Heatmap")
        self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self.canvas.draw_idle()

    def save_cma(self):
        if self.cma is None:
            QMessageBox.warning(self, "Nothing to Save", "Compute CMAs first.")
            return
        save_cma_json(self.cma, path="data/cma.json")
        QMessageBox.information(self, "Saved", "CMAs saved to data/cma.json.\nPortfolio Builder → Reload CMAs to apply.")

    def load_cma(self):
        cma = load_cma_json(path="data/cma.json")
        if cma is None:
            QMessageBox.warning(self, "No File", "No existing data/cma.json found.")
            return
        self.cma = cma
        self.render_outputs()
        self.status.setText(f"Loaded CMAs as of {cma.meta.get('as_of','?')}.")
