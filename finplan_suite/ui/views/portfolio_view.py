# finplan_suite/ui/views/portfolio_view.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QFrame, QSlider, QGridLayout, QMessageBox
)
from PyQt6.QtCore import Qt

import os, json
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


from ...core.cma import load_cma_json, CMA
from ...core.portfolio import build_frontier
from ...core.store import load_client, list_clients

RISK_TO_WEIGHTS = {
    "Conservative":            {"VOO":0.20,"IJR":0.05,"NEAR":0.20,"BOND":0.35,"TLT":0.10,"QLEIX":0.05,"VEMBX":0.04,"PDBC":0.01},
    "Moderately Conservative": {"VOO":0.30,"IJR":0.07,"NEAR":0.10,"BOND":0.35,"TLT":0.10,"QLEIX":0.04,"VEMBX":0.03,"PDBC":0.01},
    "Moderate":                {"VOO":0.40,"IJR":0.10,"NEAR":0.05,"BOND":0.28,"TLT":0.07,"QLEIX":0.05,"VEMBX":0.04,"PDBC":0.01},
    "Moderately Aggressive":   {"VOO":0.50,"IJR":0.12,"NEAR":0.03,"BOND":0.22,"TLT":0.05,"QLEIX":0.05,"VEMBX":0.02,"PDBC":0.01},
    "Aggressive":              {"VOO":0.58,"IJR":0.15,"NEAR":0.02,"BOND":0.16,"TLT":0.03,"QLEIX":0.04,"VEMBX":0.01,"PDBC":0.01},
}

def port_stats(w, mu, cov):
    ret = float(mu @ w)
    vol = float(np.sqrt(w @ cov @ w))
    return ret, vol

class PortfolioView(QWidget):
    RISK_TO_WEIGHTS = {
    "Conservative":            {"VOO":0.20,"IJR":0.05,"NEAR":0.20,"BOND":0.35,"TLT":0.10,"QLEIX":0.05,"VEMBX":0.04,"PDBC":0.01},
    "Moderately Conservative": {"VOO":0.30,"IJR":0.07,"NEAR":0.10,"BOND":0.35,"TLT":0.10,"QLEIX":0.04,"VEMBX":0.03,"PDBC":0.01},
    "Moderate":                {"VOO":0.40,"IJR":0.10,"NEAR":0.05,"BOND":0.28,"TLT":0.07,"QLEIX":0.05,"VEMBX":0.04,"PDBC":0.01},
    "Moderately Aggressive":   {"VOO":0.50,"IJR":0.12,"NEAR":0.03,"BOND":0.22,"TLT":0.05,"QLEIX":0.05,"VEMBX":0.02,"PDBC":0.01},
    "Aggressive":              {"VOO":0.58,"IJR":0.15,"NEAR":0.02,"BOND":0.16,"TLT":0.03,"QLEIX":0.04,"VEMBX":0.01,"PDBC":0.01},
}
    def __init__(self):
        super().__init__()
        self.setObjectName("PortfolioView")

        self.cma = self.load_cma_or_warn()
        self.result = None
        self.current_w = None  # current slider-based weights (normalized)

        # near the top of __init__, before creating buttons
        self.tickers = ["VOO","IJR","NEAR","BOND","TLT","QLEIX","VEMBX","PDBC"]

        root = QVBoxLayout(self)
        header = QLabel("Portfolio Builder — Efficient Frontier")
        header.setStyleSheet("font-size:20px; font-weight:600;")
        root.addWidget(header)

        # --- Top buttons ---
        top = QHBoxLayout()
        self.btn_run = QPushButton("Compute Frontier")
        self.btn_run.clicked.connect(self.run_frontier)
        top.addWidget(self.btn_run)

        self.btn_reload = QPushButton("Reload CMAs")
        self.btn_reload.clicked.connect(self.reload_cmas)
        top.addWidget(self.btn_reload)

        self.btn_preset_ms = QPushButton("Use Max Sharpe Weights")
        self.btn_preset_ms.clicked.connect(self.use_max_sharpe)
        self.btn_preset_ms.setEnabled(False)
        top.addWidget(self.btn_preset_ms)

        self.btn_preset_eq = QPushButton("Equal Weight")
        self.btn_preset_eq.clicked.connect(self.use_equal_weight)
        top.addWidget(self.btn_preset_eq)

        top.addStretch(1)

        self.btn_send_mc = QPushButton("Send to Monte Carlo")
        self.btn_send_mc.clicked.connect(self.send_to_mc)
        self.btn_send_mc.setEnabled(False)
        top.addWidget(self.btn_send_mc)

        root.addLayout(top)

        # --- Chart ---
        self.fig = Figure(figsize=(6,4), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas, 2)

        # --- Current portfolio stats ---
        stats_row = QHBoxLayout()
        self.lbl_stats = QLabel("Current: —")
        self.lbl_stats.setStyleSheet("color:#333;")
        stats_row.addWidget(self.lbl_stats)
        stats_row.addStretch(1)
        root.addLayout(stats_row)

        # --- Sliders panel ---
        sliders_frame = QFrame(self)
        sliders_layout = QGridLayout(sliders_frame)
        sliders_layout.setHorizontalSpacing(16)
        sliders_layout.setVerticalSpacing(6)

        # init storage BEFORE first use
        # (you already set this earlier; keep only one definition in __init__)
        # self.tickers = ["VOO","IJR","NEAR","BOND","TLT","QLEIX","VEMBX","PDBC"]

        self.sliders = []  # list of tuples: (QSlider, QLabelPercent)

        # header row
        sliders_layout.addWidget(QLabel("<b>Asset</b>"), 0, 0)
        sliders_layout.addWidget(QLabel("<b>Weight</b>"), 0, 1)
        sliders_layout.addWidget(QLabel("<b>%</b>"),     0, 2)

        # build rows
        for i, t in enumerate(self.tickers):
            name = QLabel(t)
            sld  = QSlider(Qt.Orientation.Horizontal)
            sld.setRange(0, 100)
            sld.setSingleStep(1)
            sld.setValue(0)
            pct  = QLabel("0%")

            # connect AFTER creating pct so handler can update it
            # we’ll use a tiny wrapper to pass index
            def _mk_handler(idx):
                return lambda val: self._on_slider_changed(idx, val)
            sld.valueChanged.connect(_mk_handler(i))

            # store tuple for downstream methods
            self.sliders.append((sld, pct))

            r = i + 1
            sliders_layout.addWidget(name, r, 0)
            sliders_layout.addWidget(sld,  r, 1)
            sliders_layout.addWidget(pct,  r, 2)

        # add the frame to your main layout
        root.addWidget(sliders_frame)


        self.btn_suggest = QPushButton("Suggest from Risk")
        self.btn_suggest.clicked.connect(self.suggest_from_risk)
        top.addWidget(self.btn_suggest)

        # --- Weights table ---
        self.tbl = QTableWidget(0, 2, self)
        self.tbl.setHorizontalHeaderLabels(["Ticker", "Weight"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.tbl, 1)

        hint = QLabel("Tip: Adjust sliders to set a custom portfolio. Weights auto-normalize to 100%.")
        hint.setStyleSheet("color:#555;")
        root.addWidget(hint)

        # initial compute
        self.run_frontier()

    # ---------- frontier & plotting ----------
    def run_frontier(self):
        mu = self.cma.exp_returns.copy()
        cov = self.cma.cov.copy()
        rf  = self.cma.rf

        # Optional per-asset cap: 50%
        w_max = 0.50

        self.result = build_frontier(mu, cov, rf=rf, k=40, w_max=w_max)
        self.btn_preset_ms.setEnabled(True)
        self.btn_send_mc.setEnabled(True)

        # set sliders to max sharpe by default
        self.set_weights(self.result.max_sharpe_w)
        self.plot_frontier()

    def plot_frontier(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # efficient frontier
        ax.plot(self.result.risks, self.result.returns, "o-", alpha=0.7, label="Efficient Frontier")

        # max sharpe marker
        ax.scatter([self.result.max_sharpe_risk], [self.result.max_sharpe_ret],
                   s=90, marker="*", label=f"Max Sharpe (SR={self.result.max_sharpe_sr:.2f})")

        # current portfolio marker
        if self.current_w is not None:
            c_ret, c_risk = port_stats(self.current_w, self.cma.exp_returns, self.cma.cov)
            ax.scatter([c_risk], [c_ret], s=70, marker="D", label="Current (sliders)")

        ax.set_xlabel("Risk (stdev)")
        ax.set_ylabel("Expected Return")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        self.canvas.draw_idle()

        # update stats label
        if self.current_w is not None:
            c_ret, c_risk = port_stats(self.current_w, self.cma.exp_returns, self.cma.cov)
            self.lbl_stats.setText(f"Current: E[Ret]={c_ret:.2%}, Risk={c_risk:.2%}")
        else:
            self.lbl_stats.setText("Current: —")

    # ---------- sliders & weights ----------
    def on_slider_change(self, _=None):
        raw = np.array([s.value() for s, _ in self.sliders], dtype=float)
        if raw.sum() <= 0:
            # avoid divide-by-zero; keep previous weights
            return
        w = raw / raw.sum()
        self.current_w = w
        self.update_slider_labels()
        self.fill_weights_table()
        self.plot_frontier()

    def set_weights(self, w):
        """Set sliders from a weight vector (auto-normalized)."""
        w = np.clip(np.array(w, dtype=float), 0, 1)
        if w.sum() == 0:
            w = np.ones_like(w) / len(w)
        w = w / w.sum()
        for (s, _), wi in zip(self.sliders, w):
            s.blockSignals(True)
            s.setValue(int(round(wi * 100)))
            s.blockSignals(False)
        self.current_w = w
        self.update_slider_labels()
        self.fill_weights_table()
        self.plot_frontier()

    def update_slider_labels(self):
        for (s, pct) in self.sliders:
            pct.setText(f"{s.value():d}%")

    def use_max_sharpe(self):
        self.set_weights(self.result.max_sharpe_w)

    def use_equal_weight(self):
        n = len(self.cma.tickers)
        self.set_weights(np.ones(n) / n)

    # ---------- table ----------
    def fill_weights_table(self):
        if self.current_w is None:
            return
        w = self.current_w
        tickers = self.cma.tickers
        self.tbl.setRowCount(len(tickers))
        for i, (tkr, wi) in enumerate(zip(tickers, w)):
            self.tbl.setItem(i, 0, QTableWidgetItem(tkr))
            self.tbl.setItem(i, 1, QTableWidgetItem(f"{wi:0.2%}"))

    def load_cma_or_warn(self) -> CMA:
        cma = load_cma_json(path="data/cma.json")
        if cma is None:
            from ...core.cma import derive_cma_from_macro
            cma = derive_cma_from_macro(gdp=0.017, cpi=0.025, real_short=0.01, term_premium=0.015)
        return cma
    
    def reload_cmas(self):
        self.cma = self.load_cma_or_warn()
        self.run_frontier()

    # ---------- send to Monte Carlo ----------
    def send_to_mc(self):
        if self.current_w is None:
            QMessageBox.warning(self, "No Weights", "Please compute or set a portfolio first.")
            return
        payload = {
            "tickers": self.cma.tickers,
            "weights": [float(x) for x in self.current_w],
            "expected_return": float(self.cma.exp_returns @ self.current_w),
            "risk": float(np.sqrt(self.current_w @ self.cma.cov @ self.current_w)),
        }
        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", "selected_portfolio.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        QMessageBox.information(self, "Sent", f"Saved selected portfolio to {path}")
    def suggest_from_risk(self):
        """Load most recently updated client, read risk band, push policy weights to sliders."""
        clients = list_clients()
        if not clients:
            QMessageBox.warning(self, "No client", "Create a client and score risk in the Risk tab first.")
            return

        c = clients[0]  # list_clients() already sorted newest-first
        band = c.risk_band or "Moderate"
        target = self.RISK_TO_WEIGHTS.get(band, self.RISK_TO_WEIGHTS["Moderate"])

        # Ensure we have tickers and sliders on this view
        if not hasattr(self, "tickers") or not hasattr(self, "sliders"):
            QMessageBox.warning(self, "Not ready", "Sliders/tickers not initialized in this view.")
            return

        # Push weights into the UI sliders (assumes sliders are 0–100 ints)
        for i, t in enumerate(self.tickers):
            w = float(target.get(t, 0.0))
            try:
                self.sliders[i].setValue(int(round(w * 100)))
            except Exception:
                pass

        # Recompute / refresh chart
        if hasattr(self, "on_slider_change"):
            self.on_slider_change()
        elif hasattr(self, "run_frontier"):
            self.run_frontier()

        QMessageBox.information(self, "Suggested Allocation",
                             f"Applied {band} policy weights for client {c.first_name} {c.last_name}.")

        