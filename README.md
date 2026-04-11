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

## Table of contents

1. [Getting started: prerequisites](#getting-started-prerequisites)
2. [GitHub: account, repository, and cloning](#github-account-repository-and-cloning)
3. [Clone this project and open it locally](#clone-this-project-and-open-it-locally)
4. [How to run the code](#how-to-run-the-code)

---

## Getting started: prerequisites

You will need a few free tools. None of this replaces reading the rest of the guide, but it is the checklist most people follow before the first successful run.

### What you will have when you are done

| Piece | Why it matters |
|------|----------------|
| **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (recommended) or local Python | A reliable way to run the same Python version and libraries this repo expects. |
| A **code editor** with a terminal | Open files, paste API keys, and run commands in one place. |
| **[Git](https://git-scm.com/downloads)** | Download this repository from GitHub (**clone**). |
| A **[GitHub](https://github.com)** account (optional but common) | Fork or star the repo, open issues, or contribute. |
| An **[Alpaca](https://alpaca.markets/)** account | Market data and (paper) trading API keys used by the fetcher and live bot. |

After you finish [How to run the code](#how-to-run-the-code) you will be able to:

1. Download historical prices for the default **Nasdaq-100** universe (plus the regime ETF **QQQ** when enabled in config).
2. Run the **backtest** and open a results dashboard in your browser (default port **8063**).
3. Optionally run the **live bot** (paper trading recommended) and the **live dashboard** (default port **8066**).

### Docker Desktop (recommended)

Docker runs a small **Linux environment** on your computer so you can use the exact same `python` and `pip` steps as everyone else, without fighting OS-specific Python installs.

1. Open **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** and download it for **Windows** or **Mac** (pick Apple Silicon or Intel when asked).
2. Run the installer and **restart** if the installer tells you to.
3. Launch **Docker Desktop** and wait until it reports **running** (whale icon in the system tray on Windows or the menu bar on Mac).

**Windows note:** Docker may ask you to enable **WSL 2**. That is normal; follow the prompts in the Docker installer or see [Microsoft’s WSL install guide](https://learn.microsoft.com/en-us/windows/wsl/install).

**Linux:** You usually install **[Docker Engine](https://docs.docker.com/engine/install/)** instead of Docker Desktop (same idea for this project).

**Important:** This repository does **not** ship a custom image. You will start an official **`python:3.12-bookworm`** container and install the project inside it ([How to run the code](#how-to-run-the-code)). That is intentional and keeps the README simple.

### Code editor and terminal

**Recommended: [Cursor](https://cursor.com)** — a desktop editor built like VS Code, with a built-in AI assistant and a **terminal** where you run `git`, `docker`, and `python`. Download, install, then use **File → Open Folder** to open the cloned project.

**Alternatives** (any of these work if you prefer them):

- **[Visual Studio Code](https://code.visualstudio.com/)** — the widely used free editor Cursor is based on. Install the **Python** extension if prompted. You can add **[GitHub Copilot](https://github.com/features/copilot)** or other assistants separately.
- **[Windsurf](https://windsurf.com/)** and other **VS Code–compatible** editors — same idea: open the project folder and use the integrated terminal.
- **[VSCodium](https://vscodium.com/)** — an open-source build of VS Code without Microsoft branding (optional).

Whatever you pick, you need a **terminal inside the project folder** for the commands in [How to run the code](#how-to-run-the-code). In Cursor and VS Code that is usually **Terminal → New Terminal** (or the shortcut shown in the app).

### Git on your computer

**Git** is the command-line tool that talks to GitHub and **clones** (downloads) repositories.

- **Windows:** Install from **[git-scm.com/download/win](https://git-scm.com/download/win)** (default options are fine).
- **Mac:** Run `xcode-select --install` in **Terminal** for Apple’s command-line tools, or install Git from **[git-scm.com](https://git-scm.com/downloads)**.
- **Linux:** Install `git` with your package manager (for example `sudo apt install git` on Debian/Ubuntu).

Check that it works:

```bash
git --version
```

You should see a version number.

---

## GitHub: account, repository, and cloning

**[GitHub](https://github.com)** hosts this project as a **repository** (“repo”): a folder with code and history. You **browse** it in the browser; you **clone** it to your machine to run it.

### Create an account (if you do not have one)

1. Go to **[github.com/signup](https://github.com/signup)** and create a free account.
2. Verify your email if GitHub asks you to.

You can **clone** public repositories without an account, but having an account lets you **fork** (your own copy), star repos, and open issues.

### Fork vs clone (short version)

- **Clone** — copies the repo **to your computer** so you can run code. You will do this in the next section.
- **Fork** — creates **your copy of the repo on GitHub** (useful if you plan to change code and open pull requests). After forking, clone **your** fork’s URL instead of the original.

### Where to copy the clone URL

1. Open the repository page on GitHub (for example `https://github.com/OWNER/REPO_NAME`).
2. Click the green **Code** button.
3. Under **HTTPS**, copy the URL (it looks like `https://github.com/OWNER/REPO_NAME.git`).

Official GitHub help: **[Cloning a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)**.

---

## Clone this project and open it locally

1. **Pick a parent folder** on your disk where projects live (for example `Documents\Projects` on Windows or `~/Projects` on Mac). You do **not** need to create the repo folder yourself; `git clone` will create it.
2. Open your editor (**Cursor**, **VS Code**, or similar) and open a terminal (**Terminal → New Terminal**).
3. In the terminal, `cd` to that parent folder, then clone (paste **your** URL from GitHub’s green **Code** button):

```bash
cd ~/Projects
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

If the folder name contains spaces (for example `DeepVibe AI Hedge Fund`), quote the path when you `cd`:

```bash
cd "/path/to/DeepVibe AI Hedge Fund"
```

4. **Open the project in your editor:** **File → Open Folder** and select the **repository root** (the folder that contains `README.md` and `pyproject.toml`). From now on, every command in this README assumes the terminal’s current directory is that root.

---

## How to run the code

The steps below pick up after the repository is open on your machine. Complete them **in order**.

### 1. Run a Python environment (choose one)

#### Option A — Docker (recommended for beginners)

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

#### Option B — Python on your computer

1. Install **Python 3.10 or newer** from **[python.org/downloads](https://www.python.org/downloads/)** (on Windows, enable **Add Python to PATH** if the installer offers it).
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

### 2. Install project dependencies

In the **same** terminal (Docker shell or activated venv), from the **project root**:

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

Wait until it finishes without errors.

---

### 3. Tell Python where the code lives (PYTHONPATH)

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

### 4. Alpaca API keys and safety (paper trading)

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

### 5. Default universe: Nasdaq-100

You do **not** need to edit anything for your **first** run. In `config.py`, **`MAD_UNIVERSE_TICKERS`** is set to **`nasdaq100`**, and the pipeline mode loads that full list (plus the **QQQ** regime ticker when regime logic is enabled).

The first **download** can take a long time (many symbols and years of daily data). Let it finish; interrupting can leave partial data.

---

### 6. First-time data pipeline (required before backtest)

Run **6a** and **6b** **in order** from the project root, with `PYTHONPATH` set as in [step 3](#3-tell-python-where-the-code-lives-pythonpath). Step **6c** is optional.

#### 6a. Download daily prices (Alpaca to SQLite)

```bash
python -m deepvibe_hedge.alpaca_fetcher
```

This creates files under **`data/ohlcv/`** (that folder is not stored in Git).

#### 6b. Walk-forward splits and moving averages

```bash
python -m deepvibe_hedge.data_splitter
```

This updates the same databases with split labels and SMA columns used by the backtester and live stack.

#### 6c. (Optional) Inspect data

```bash
python -m deepvibe_hedge.db_utils
```

---

### 7. Run the default backtest (Nasdaq-100 panel)

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

### 8. Live bot (Alpaca execution)

**Only after** steps [6a](#6a-download-daily-prices-alpaca-to-sqlite)–[6b](#6b-walk-forward-splits-and-moving-averages) have succeeded and you are on **paper** keys unless you fully understand live risk:

#### First time (or starting mid-day)

The long-running bot **waits until after the official NYSE close** for its automatic rebalance. If you turn it on earlier in the day (for example **2:00 p.m. US/Eastern**), it will **not** place that day’s strategy trades until after the bell.

To **open or align positions right away** before you leave the bot running:

1. Run the **one-shot** helper (same as ``live_bot`` with ``--once``):

```bash
python -m deepvibe_hedge.mad.one_time_portfolio_rebalance
```

2. Then start the **long-running** bot so it can handle **future** session closes:

```bash
python -m deepvibe_hedge.mad.live_bot
```

Optional: ``--dry-run`` works on the helper too (targets only, no orders):

```bash
python -m deepvibe_hedge.mad.one_time_portfolio_rebalance --dry-run
```

#### Commands (reference)

```bash
# See targets only — no orders
python -m deepvibe_hedge.mad.live_bot --dry-run

# One immediate reconcile cycle (good for cron)
python -m deepvibe_hedge.mad.live_bot --once

# Long-running: wakes periodically and trades once per NYSE session after the official close
python -m deepvibe_hedge.mad.live_bot
```

Printed times use **US/Eastern**. The bot can **append** new daily bars to your SQLite files before each cycle when configured (see `config.py` and the module docstring in `live_bot.py`).

**Post-close window:** **`MAD_LIVE_REBALANCE_WINDOW_MINUTES`** (default **90**) is how many minutes **after Alpaca’s official session close** the bot is allowed to run that day’s **single** automated rebalance. Keeps your **one** batch of per-symbol orders in the **early** after-hours period (extended-hours limits still apply), instead of firing late at night if the process only wakes then. Set to **0** for legacy behavior (any time after close). Use **`--once`** for a manual rebalance outside the window.

---

### 9. Live dashboard (equity, MRAT chart, watchlist)

In a **second** terminal (same `PYTHONPATH=src`, same project root):

```bash
python -m deepvibe_hedge.mad.live_dashboard
```

Then open:

**http://127.0.0.1:8066**

The first MRAT panel build can take **several minutes** while it reads many SQLite files; the equity section usually appears first. The app binds to **`0.0.0.0`** so other machines on your LAN can open it if your firewall allows (be careful on untrusted networks).

---

### 10. If something goes wrong

| Symptom | What to check |
|--------|----------------|
| `ModuleNotFoundError` | `PYTHONPATH=src` (or PowerShell `$env:PYTHONPATH="src"`) in **this** terminal session ([step 3](#3-tell-python-where-the-code-lives-pythonpath)). |
| Alpaca errors | `.env` keys, **paper vs live** keys matching `BOT_MODE` in `config.py`. |
| Backtest says not enough tickers | You need enough symbols with OHLCV databases; run the **fetcher** for the full universe ([step 6a](#6a-download-daily-prices-alpaca-to-sqlite)). |
| Docker volume empty on Windows | Use **PowerShell** and `${PWD}` as in [step 1](#1-run-a-python-environment-choose-one), and ensure Docker Desktop file sharing includes your drive. |
| Port already in use | Change **`MAD_DASHBOARD_PORT`** in `config.py` (backtest) or edit **`DASHBOARD_PORT`** in `mad/live_dashboard.py` (live UI). |

---

### 11. Technical reference (for developers)

| Path | Role |
|------|------|
| `src/deepvibe_hedge/config.py` | Universe, dates, splitter, MAD grids, live flags, `BOT_MODE` |
| `src/deepvibe_hedge/alpaca_fetcher.py` | Historical bars → `data/ohlcv/{SYMBOL}_{gran}.db` + `.csv` |
| `src/deepvibe_hedge/data_splitter.py` | Splits + SMA columns |
| `src/deepvibe_hedge/ohlcv_live_append.py` | Live incremental daily bars + SMA refresh |
| `src/deepvibe_hedge/db_utils.py` | CLI to inspect OHLCV SQLite files |
| `src/deepvibe_hedge/mad/backtester.py` | Panel backtest, optimiser SQLite under `data/mad/` |
| `src/deepvibe_hedge/mad/live_bot.py` | Alpaca paper/live execution |
| `src/deepvibe_hedge/mad/one_time_portfolio_rebalance.py` | Wrapper: immediate reconcile (``live_bot --once``) |
| `src/deepvibe_hedge/mad/live_dashboard.py` | Dash live UI (dark theme in `mad/dash_assets/theme.css`) |
| `src/deepvibe_hedge/paths.py` | `DATA_ROOT`, `OHLCV_DIR`, `MAD_DATA_DIR` |

**Strategy summary:** MRAT ranks tickers by short-SMA / long-SMA vs the cross-section; σ-bands and deciles set long/short rules. Optional **regime** uses an ETF (default **QQQ**) to reduce exposure when the market is below a long moving average. Details: docstrings in `mad/backtester.py` and `config.py`.

**Related:** `mad/walkforward_oos.py`, `mad/permutation_test.py`; `reference_old_folder/` is legacy only.

---

## Logo

Brand asset: **`deepvibe2.png`** at the repository root (paths in this README are relative so images render on GitHub/GitLab).
