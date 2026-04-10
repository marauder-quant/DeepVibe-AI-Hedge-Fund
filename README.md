<p align="center">
  <img src="deepvibe2.png" alt="DeepVibe AI Hedge Fund" width="480">
</p>

<p align="center">
  <img src="Deepvibe_results_backtest.PNG" alt="DeepVibe backtest results" width="900">
</p>

# DeepVibe AI Hedge Fund

Standalone **MAD / MRAT** stack: download daily stock prices from **Alpaca**, store them in **SQLite**, run a **panel backtest** on your universe (default **Nasdaq-100**), and optionally trade the same logic with the **live bot** and watch it on the **live dashboard**.

This README is written for **new users** who are comfortable clicking and typing commands, but **do not need prior coding experience**. Follow the sections in order.

---

## What you will install and run

| Step | What it is |
|------|------------|
| **Docker** (optional but recommended) | A “boxed” Linux environment so Python and libraries work the same on any computer. |
| **Cursor** | A free editor (like VS Code) with a built-in terminal where you run commands. |
| **Git** | Used once to **clone** (download) this repository. |
| **Alpaca account** | Free signup for market data and paper trading API keys. |

After setup you will:

1. Download historical prices for the default **Nasdaq-100** universe (plus the regime ETF **QQQ** if enabled in config).
2. Run the **backtest** and open a results dashboard in your browser (default port **8063**).
3. Optionally run the **live bot** (paper trading recommended) and the **live dashboard** (default port **8066**).

---

## 1. Install Docker Desktop

Docker lets you run a standard **Python** environment without installing Python directly on Windows or macOS.

1. Open **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** in your browser.
2. Download Docker Desktop for **Windows** or **Mac** (Apple Silicon or Intel as prompted).
3. Run the installer and **restart** your computer if asked.
4. Start **Docker Desktop** from your applications menu and wait until it says it is **running** (whale icon in the system tray or menu bar).

**Linux:** Install Docker Engine using your distribution’s instructions ([Docker Engine on Linux](https://docs.docker.com/engine/install/)).

**Note:** This repository does not ship a custom Docker image. You will use the official **Python** image and install this project inside it (steps below). That is normal.

---

## 2. Install Cursor

1. Open **[cursor.com](https://cursor.com)** and download **Cursor** for your system.
2. Install it like any other application.
3. Launch Cursor. You can sign in or skip if you prefer.

You will use Cursor to **open the project folder** and run commands in its **terminal** (menu: **Terminal → New Terminal**, or the shortcut shown in the app).

---

## 3. Install Git (if you do not have it)

- **Windows:** Install from **[git-scm.com](https://git-scm.com/download/win)** (default options are fine).
- **Mac:** Install **Xcode Command Line Tools** (`xcode-select --install` in Terminal) or use the Git installer from git-scm.com.

Check that Git works: open a terminal and run:

```bash
git --version
```

You should see a version number.

---

## 4. Clone this repository

1. In Cursor, use **File → Open Folder** and pick an empty parent folder where you want the project (for example `Documents\Projects` on Windows or `~/Projects` on Mac).
2. Open the **terminal** in Cursor (**Terminal → New Terminal**).
3. Clone the repo (replace the URL with **your** fork or the URL your team gave you):

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

If your folder name contains spaces (for example `DeepVibe AI Hedge Fund`), keep the quotes when you `cd`:

```bash
cd "DeepVibe AI Hedge Fund"
```

---

## 5. Run a Python environment (choose one)

### Option A — Docker (recommended for beginners)

From your **project root** (the folder that contains `README.md` and `pyproject.toml`), run:

**Mac or Linux:**

```bash
docker run -it --rm -v "$PWD:/workspace" -w /workspace python:3.12-bookworm bash
```

**Windows (PowerShell),** from the project folder:

```powershell
docker run -it --rm -v "${PWD}:/workspace" -w /workspace python:3.12-bookworm bash
```

You should see a Linux prompt. **Stay inside this container** for all `pip` and `python` commands until you type `exit`.

### Option B — Python on your computer

1. Install **Python 3.10 or newer** from **[python.org](https://www.python.org/downloads/)** (check “Add Python to PATH” on Windows).
2. In the project folder, create a virtual environment and activate it:

**Mac / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## 6. Install project dependencies

In the **same** terminal (Docker shell or activated venv), from the **project root**:

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

Wait until it finishes without errors.

---

## 7. Tell Python where the code lives (`PYTHONPATH`)

This project keeps its package under `src/`. Every time you open a **new** terminal, set:

**Mac / Linux:**

```bash
export PYTHONPATH=src
```

**Windows PowerShell:**

```powershell
$env:PYTHONPATH = "src"
```

**Tip:** If your path has spaces, always `cd` into the folder with quotes, for example:

```bash
cd "/path/to/DeepVibe AI Hedge Fund"
export PYTHONPATH=src
```

---

## 8. Alpaca API keys and safety (paper trading)

1. Create a free account at **[Alpaca](https://alpaca.markets/)** and open the **dashboard** for API keys.
2. Copy **Paper Trading** keys first (recommended for learning).
3. In the project root, copy the example env file:

```bash
cp .env.example .env
```

4. Open **`.env`** in Cursor and paste your keys. You can use either the generic names or the paper-specific ones (see comments inside `.env.example`).

5. Open **`src/deepvibe_hedge/config.py`** in Cursor and find **`BOT_MODE`**. For learning, set:

```python
BOT_MODE = "paper"
```

**Never commit or share your `.env` file.** It is ignored by Git in normal setups.

---

## 9. Default universe: Nasdaq-100

You do **not** need to edit anything for your **first** run. In `config.py`, **`MAD_UNIVERSE_TICKERS`** is set to **`nasdaq100`**, and the pipeline mode loads that full list (plus the **QQQ** regime ticker when regime logic is enabled).

The first **download** can take a long time (many symbols and years of daily data). Let it finish; interrupting can leave partial data.

---

## 10. First-time data pipeline (required before backtest)

Run **10a** and **10b** **in order** from the project root, with `PYTHONPATH` set as in section 7. Step **10c** is optional.

### 10a. Download daily prices (Alpaca → SQLite)

```bash
python -m deepvibe_hedge.alpaca_fetcher
```

This creates files under **`data/ohlcv/`** (that folder is not stored in Git).

### 10b. Walk-forward splits and moving averages

```bash
python -m deepvibe_hedge.data_splitter
```

This updates the same databases with split labels and SMA columns used by the backtester and live stack.

### 10c. (Optional) Inspect data

```bash
python -m deepvibe_hedge.db_utils
```

---

## 11. Run the default backtest (Nasdaq-100 panel)

```bash
python -m deepvibe_hedge.mad.backtester
```

- The terminal will print progress and metrics.
- By default it also starts a **Dash** dashboard for exploring results.

Open a browser and go to:

**http://127.0.0.1:8063**

(Port **8063** comes from **`MAD_DASHBOARD_PORT`** in `config.py`.)

To run **without** opening the dashboard (terminal only):

```bash
python -m deepvibe_hedge.mad.backtester --no-dashboard
```

Output databases and tables are written under **`data/mad/`** (paths depend on reference ticker and bar size, for example `QQQ_1d_mad_optim.db`).

---

## 12. Live bot (Alpaca execution)

**Only after** steps 10a–10b have succeeded and you are on **paper** keys unless you fully understand live risk:

```bash
# See targets only — no orders
python -m deepvibe_hedge.mad.live_bot --dry-run

# One immediate reconcile cycle (good for cron)
python -m deepvibe_hedge.mad.live_bot --once

# Long-running: wakes periodically and trades once per NYSE session after the official close
python -m deepvibe_hedge.mad.live_bot
```

Printed times use **US/Eastern**. The bot can **append** new daily bars to your SQLite files before each cycle when configured (see `config.py` and the module docstring in `live_bot.py`).

---

## 13. Live dashboard (equity, MRAT chart, watchlist)

In a **second** terminal (same `PYTHONPATH=src`, same project root):

```bash
python -m deepvibe_hedge.mad.live_dashboard
```

Then open:

**http://127.0.0.1:8066**

The first MRAT panel build can take **several minutes** while it reads many SQLite files; the equity section usually appears first. The app binds to **`0.0.0.0`** so other machines on your LAN can open it if your firewall allows (be careful on untrusted networks).

---

## 14. If something goes wrong

| Symptom | What to check |
|--------|----------------|
| `ModuleNotFoundError` | `PYTHONPATH=src` (or PowerShell `$env:PYTHONPATH="src"`) in **this** terminal session. |
| Alpaca errors | `.env` keys, **paper vs live** keys matching `BOT_MODE` in `config.py`. |
| Backtest says not enough tickers | You need enough symbols with OHLCV databases; run the **fetcher** for the full universe (section 10a). |
| Docker volume empty on Windows | Use **PowerShell** and `${PWD}` as in section 5, and ensure Docker Desktop file sharing includes your drive. |
| Port already in use | Change **`MAD_DASHBOARD_PORT`** in `config.py` (backtest) or edit **`DASHBOARD_PORT`** in `mad/live_dashboard.py` (live UI). |

---

## 15. Technical reference (for developers)

| Path | Role |
|------|------|
| `src/deepvibe_hedge/config.py` | Universe, dates, splitter, MAD grids, live flags, `BOT_MODE` |
| `src/deepvibe_hedge/alpaca_fetcher.py` | Historical bars → `data/ohlcv/{SYMBOL}_{gran}.db` + `.csv` |
| `src/deepvibe_hedge/data_splitter.py` | Splits + SMA columns |
| `src/deepvibe_hedge/ohlcv_live_append.py` | Live incremental daily bars + SMA refresh |
| `src/deepvibe_hedge/db_utils.py` | CLI to inspect OHLCV SQLite files |
| `src/deepvibe_hedge/mad/backtester.py` | Panel backtest, optimiser SQLite under `data/mad/` |
| `src/deepvibe_hedge/mad/live_bot.py` | Alpaca paper/live execution |
| `src/deepvibe_hedge/mad/live_dashboard.py` | Dash live UI (dark theme in `mad/dash_assets/theme.css`) |
| `src/deepvibe_hedge/paths.py` | `DATA_ROOT`, `OHLCV_DIR`, `MAD_DATA_DIR` |

**Strategy summary:** MRAT ranks tickers by short-SMA / long-SMA vs the cross-section; σ-bands and deciles set long/short rules. Optional **regime** uses an ETF (default **QQQ**) to reduce exposure when the market is below a long moving average. Details: docstrings in `mad/backtester.py` and `config.py`.

**Related:** `mad/walkforward_oos.py`, `mad/permutation_test.py`; `reference_old_folder/` is legacy only.

---

## Logo

Brand asset: **`deepvibe2.png`** at the repository root (paths in this README are relative so images render on GitHub/GitLab).
