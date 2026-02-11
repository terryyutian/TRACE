# TRACE
TRACE: An Automated Approach to Text Revision Analysis via Change Estimation

TRACE is an automated approach for multi-granularity analysis of textual revisions between a source text and a target text.

The current repo contains the code for this approach as well as a demo web application to use TRACE for text revision analysis. 

Below are instructions about how to run this demo app. 

## Run the TRACE Demo Web App (FastAPI)

### 1) Prerequisites

- Python: **3.10–3.12**
- `pip` installed
- (Recommended) a fresh virtual environment

### 2) Create and activate a virtual environment (recommended)

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate

```

#### Windows (PowerShell)

```bash
python -m venv .venv
source .venv/scripts/activate
```

### 3) Install TRACE (editable) + web dependencies
From the repository root (where `pyproject.toml` is located):

```bash
pip install -U pip
pip install -e .
```

### 4) Install the spaCy model
TRACE uses spaCy for sentence segmentation. Install the English model:

```bash
python -m spacy download en_core_web_sm
```

### 5) Start the web app
Run the FastAPI server with Uvicorn from the repository root:

```bash
python -m uvicorn trace_revision.app:app --reload
```

You should see output similar to:
`Uvicorn running on http://127.0.0.1:8000`

### 6) Open the app in your browser
Go to:
`http://127.0.0.1:8000`

### 7) Use the app

- Paste your draft in Source Text

- Paste your revision in Target Text

- Click Analyze Text Changes

- Review results in the table

- Click Export Results to download the CSV

No data is stored on the server; the analysis runs on demand and results are returned to your browser.


