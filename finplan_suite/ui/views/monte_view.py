# finplan_suite/ui/views/monte_view.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFormLayout, QLineEdit,
    QPushButton, QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout, QTableWidget,
    QTableWidgetItem, QMessageBox
)
from PyQt6.QtCore import Qt

import os, json, datetime
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ...core.cma import load_cma_json, derive_cma_from_macro, CMA
from ...core.monte_carlo import MCInputs, simulate_paths
from ...core.store import list_clients

def fmt_money(x):  # simple formatter
    return f"${x:,.0f}"

class MonteCarloView(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("MonteCarloView")

        self.cma = self.load_cma_or_default()
        self.loaded_port = None  # dict from selected_portfolio.json or None

        root = QVBoxLayout(self)

        title = QLabel("Monte Carlo & Cashflows")
        title.setStyleSheet("font-size:20px; font-weight:600;")
        root.addWidget(title)

        desc = QLabel("Simulate portfolio value with savings before retirement and withdrawals after, including pension, Social Security, and inflation.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#555;")
        root.addWidget(desc)

        # --- Inputs panel ---
        form_box = QGroupBox("Inputs")
        form = QGridLayout(form_box)

        # Left column: balances & flows
        self.in_init = QDoubleSpinBox();   self.in_init.setRange(0, 1e9);   self.in_init.setValue(500000); self.in_init.setDecimals(2)
        self.in_contrib = QDoubleSpinBox();self.in_contrib.setRange(0, 1e7); self.in_contrib.setValue(30000); self.in_contrib.setDecimals(2)
        self.in_years = QSpinBox();        self.in_years.setRange(5, 80);    self.in_years.setValue(40)
        self.in_to_ret = QSpinBox();       self.in_to_ret.setRange(0, 60);   self.in_to_ret.setValue(25)

        self.in_income = QDoubleSpinBox(); self.in_income.setRange(0, 1e7);  self.in_income.setValue(180000); self.in_income.setDecimals(2)
        self.in_pen = QDoubleSpinBox();    self.in_pen.setRange(0, 1e7);     self.in_pen.setValue(0); self.in_pen.setDecimals(2)
        self.in_ss = QDoubleSpinBox();     self.in_ss.setRange(0, 1e7);      self.in_ss.setValue(50000); self.in_ss.setDecimals(2)

        self.in_infl = QDoubleSpinBox();   self.in_infl.setRange(0.0, 0.15); self.in_infl.setSingleStep(0.001); self.in_infl.setValue(0.025); self.in_infl.setDecimals(4)

        # Right column: return model & trials
        self.in_mu = QDoubleSpinBox();     self.in_mu.setRange(-0.5, 0.5);   self.in_mu.setSingleStep(0.001); self.in_mu.setValue(0.07); self.in_mu.setDecimals(4)
        self.in_sigma = QDoubleSpinBox();  self.in_sigma.setRange(0.0, 1.0); self.in_sigma.setSingleStep(0.001); self.in_sigma.setValue(0.15); self.in_sigma.setDecimals(4)
        self.in_trials = QSpinBox();       self.in_trials.setRange(100, 200000); self.in_trials.setValue(10000)

        # Auto-fill from Portfolio button
        self.btn_load_port = QPushButton("Load from Portfolio Builder")
        self.btn_load_port.clicked.connect(self.load_selected_portfolio)

        # Layout form (labels, inputs)
        r = 0
        form.addWidget(QLabel("<b>Pre-Retirement</b>"), r, 0, 1, 2); r+=1
        form.addWidget(QLabel("Initial portfolio"), r, 0); form.addWidget(self.in_init, r, 1); r+=1
        form.addWidget(QLabel("Annual savings"), r, 0);   form.addWidget(self.in_contrib, r, 1); r+=1
        form.addWidget(QLabel("Years (total)"), r, 0);    form.addWidget(self.in_years, r, 1); r+=1
        form.addWidget(QLabel("Years to retirement"), r, 0); form.addWidget(self.in_to_ret, r, 1); r+=1

        form.addWidget(QLabel("<b>Retirement Cashflows</b>"), r, 0, 1, 2); r+=1
        form.addWidget(QLabel("Desired first-year income"), r, 0); form.addWidget(self.in_income, r, 1); r+=1
        form.addWidget(QLabel("Pension (year 1)"), r, 0);         form.addWidget(self.in_pen, r, 1); r+=1
        form.addWidget(QLabel("Social Security (year 1)"), r, 0); form.addWidget(self.in_ss, r, 1); r+=1
        form.addWidget(QLabel("Inflation (annual, dec)"), r, 0);  form.addWidget(self.in_infl, r, 1); r+=1

        form.addWidget(QLabel("<b>Return Model</b>"), r, 0, 1, 2); r+=1
        form.addWidget(QLabel("Expected return (μ)"), r, 0);      form.addWidget(self.in_mu, r, 1); r+=1
        form.addWidget(QLabel("Std dev (σ)"), r, 0);              form.addWidget(self.in_sigma, r, 1); r+=1
        form.addWidget(QLabel("Trials"), r, 0);                   form.addWidget(self.in_trials, r, 1); r+=1
        form.addWidget(self.btn_load_port, r, 0, 1, 2); r+=1

        root.addWidget(form_box)

        # Run button row
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.clicked.connect(self.run_sim)
        run_row.addWidget(self.btn_run)
        self.lbl_headline = QLabel("Success: —   |   Ruin: —")
        run_row.addStretch(1)
        run_row.addWidget(self.lbl_headline)
        root.addLayout(run_row)
        self.btn_load_client = QPushButton("Load from Client Profile")
        self.btn_load_client.clicked.connect(self.load_mc_defaults)
        root.addWidget(self.btn_load_client)

        # Chart
        self.fig = Figure(figsize=(6,4), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas, 2)

        # Final value percentiles
        self.tbl = QTableWidget(0, 2, self)
        self.tbl.setHorizontalHeaderLabels(["Percentile", "Final Value"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.tbl, 1)

        # Auto-load from portfolio if available
        self.load_selected_portfolio()
    def load_cma_or_default(self) -> CMA:

        cma = load_cma_json(path="data/cma.json")
        if cma is None:
        # Fallback baseline if Economic → CMA hasn't been saved yet
            cma = derive_cma_from_macro(gdp=0.017, cpi=0.025, real_short=0.010, term_premium=0.015)
        return cma
    # --------------------------
    # Load selected portfolio -> sets mu, sigma from CMA & weights
    def load_selected_portfolio(self):
        path = os.path.join("data", "selected_portfolio.json")
        if not os.path.exists(path):
            self.lbl_headline.setText("Success: —   |   Ruin: —    (Tip: set μ & σ manually or send from Portfolio Builder)")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            self.lbl_headline.setText(f"Failed to load selected portfolio: {e}")
            return

        # Map weights to current CMA order (safety)
        tickers = obj.get("tickers", [])
        weights = np.array(obj.get("weights", []), dtype=float)
        order = [self.cma.tickers.index(t) for t in tickers]
        w_full = np.zeros(len(self.cma.tickers))
        for src_idx, tgt_idx in enumerate(order):
            w_full[tgt_idx] = weights[src_idx]
        w_full = w_full / (w_full.sum() if w_full.sum() > 0 else 1.0)

        # Portfolio μ, σ from CMA
        mu = float(self.cma.exp_returns @ w_full)
        sigma = float(np.sqrt(w_full @ self.cma.cov @ w_full))

        # Fill controls
        self.in_mu.setValue(mu)
        self.in_sigma.setValue(sigma)
        self.lbl_headline.setText("Loaded portfolio. Adjust inputs and run.")

    # --------------------------
    def run_sim(self):
        p = MCInputs(
            init_value      = float(self.in_init.value()),
            annual_invest   = float(self.in_contrib.value()),
            horizon_years   = int(self.in_years.value()),
            years_to_retire = int(self.in_to_ret.value()),
            desired_income  = float(self.in_income.value()),
            pension         = float(self.in_pen.value()),
            social_security = float(self.in_ss.value()),
            inflation       = float(self.in_infl.value()),
            mu              = float(self.in_mu.value()),
            sigma           = float(self.in_sigma.value()),
            n_paths         = int(self.in_trials.value()),
            seed            = 42,
        )

        out = simulate_paths(p)

        # Headline stats
        succ = out["success_rate"] * 100.0
        ruin = out["ruin_rate"] * 100.0
        self.lbl_headline.setText(f"Success: {succ:.1f}%   |   Ruin: {ruin:.1f}%")

        # Plot fan (mean + 5–95%)
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        yrs = out["years"]
        ax.plot(yrs, out["mean_path"], "k", lw=2.5, label="Mean")
        ax.fill_between(yrs, out["p5_path"], out["p95_path"], alpha=0.22, label="5th–95th %")
        ax.set_xlabel("Years from Today")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title("Monte Carlo Simulation")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        # Friendly y formatting
        from matplotlib.ticker import StrMethodFormatter, MaxNLocator
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        self.canvas.draw_idle()

        # Final value percentiles table
        pct = out["final_value_percentiles"]
        rows = [(5, pct[5]), (25, pct[25]), (50, pct[50]), (75, pct[75]), (95, pct[95])]
        self.tbl.setRowCount(len(rows))
        for i, (pctl, val) in enumerate(rows):
            self.tbl.setItem(i, 0, QTableWidgetItem(f"{pctl}th"))
            self.tbl.setItem(i, 1, QTableWidgetItem(fmt_money(val)))

    def showEvent(self, e):
        super().showEvent(e)
        # Auto-load whenever the tab becomes visible (safe/no-op if missing data)
        try:
            self.load_mc_defaults()
        except Exception as ex:
            print(f"[MonteCarloView] auto-load warning: {ex}")

    def _years_to_model(self, age: int | None, horizon_to_age: int = 95) -> int:
        """Simple rule: model to age 95 by default; min 30 years."""
        if age is None or age <= 0:
            return 35
        return max(30, horizon_to_age - age)

    def _read_selected_portfolio(self):
        """Read data/selected_portfolio.json if it exists (from Portfolio tab)."""
        path = os.path.join("data", "selected_portfolio.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _portfolio_return_risk_fallback(self):
        """If no selected portfolio, use CMA with equal weights."""
        cma = load_cma_json(path="data/cma.json")
        if cma is None:
            return 0.05, 0.10  # mild defaults
        n = len(cma.tickers)
        w = np.array([1.0 / n] * n, dtype=float)
        mu = float(cma.exp_returns @ w)
        sig = float(np.sqrt(w @ cma.cov @ w))
        return mu, sig

    def load_mc_defaults(self):
        """
        Populate Monte Carlo inputs from:
        - Most recently updated client in data/clients/
        - Selected portfolio in data/selected_portfolio.json (if any)
        - CMA fallback for expected return/risk if selected portfolio not present
        """
        clients = list_clients()
        if not clients:
            QMessageBox.information(self, "No client", "Create & save a client first in Client Profile.")
            return

        c = clients[0]  # newest first from store.py

        # Age
        this_year = datetime.date.today().year
        age = None
        try:
            if c.birth_year:
                age = max(0, this_year - int(c.birth_year))
        except Exception:
            age = None

        # Retirement timing & income
        ret_age = c.retirement_age if c.retirement_age else 65
        yrs_to_ret = max(0, (ret_age - age)) if (age is not None and ret_age) else 10
        desired_income = float(c.retirement_spending) if c.retirement_spending not in (None, "") else 0.0

        # Initial portfolio: sum of accounts
        init_value = 0.0
        try:
            for a in (c.accounts or []):
                init_value += float(a.get("value", 0.0) or 0.0)
        except Exception:
            pass

        # Horizon
        years_total = self._years_to_model(age, 95)

        # Portfolio μ/σ from selected portfolio, or CMA fallback
        sel = self._read_selected_portfolio()
        if sel:
            exp_ret = float(sel.get("expected_return", 0.05))  # decimal
            risk    = float(sel.get("risk", 0.10))            # decimal
        else:
            exp_ret, risk = self._portfolio_return_risk_fallback()

        # Inflation (decimal) — later: wire to CPI forecast
        inflation = 0.025

        # Defaults for now (later: wire client cash flows if you capture them)
        annual_invest = 0.0
        pension = 0.0
        social  = 0.0

        # ---- Push into widgets (use setValue on spin boxes) ----
        self.in_init.setValue(float(init_value))
        self.in_contrib.setValue(float(annual_invest))
        self.in_years.setValue(int(years_total))
        self.in_to_ret.setValue(int(yrs_to_ret))
        self.in_income.setValue(float(desired_income))
        self.in_pen.setValue(float(pension))
        self.in_ss.setValue(float(social))
        self.in_infl.setValue(float(inflation))  # decimal, e.g., 0.025

        self.in_mu.setValue(float(exp_ret))      # decimal
        self.in_sigma.setValue(float(risk))      # decimal

        self.lbl_headline.setText("Loaded assumptions from Client Profile (and Portfolio/CMA).")
