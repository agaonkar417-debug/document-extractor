# Document extractor

Pull structured JSON out of **invoices** and **resumes** with an LLM, and benchmark models
side by side on **accuracy, latency and cost** in one table.

- Works with **Gemini** (REST API, no SDK needed) and **Claude** (Anthropic SDK).
- Reads `.txt` / `.md`, `.pdf`, and images (`.png`, `.jpg`, `.webp`).
- Output always matches a fixed schema (the models' structured-output / JSON mode),
  and is validated with Pydantic.
- Ships with 6 sample documents and hand-written ground truth, so you can run the benchmark
  right away and add your own documents.

## Quick start

Requires Python 3.10+.

```powershell
git clone <this repo>
cd <this repo>
pip install -r requirements.txt
```

### 1. Set your API key

Get a Gemini key at [aistudio.google.com](https://aistudio.google.com/apikey) (the free tier
works). Save it as an environment variable — never put it in a file in this repo.

Windows (PowerShell) — prompts for the key without showing it and saves it for your user:

```powershell
$k = Read-Host "Paste Gemini API key" -AsSecureString; [Environment]::SetEnvironmentVariable("GEMINI_API_KEY", [Runtime.InteropServices.Marshal]::PtrToStringBSTR([Runtime.InteropServices.Marshal]::SecureStringToBSTR($k)), "User")
```

Open a **new** terminal afterwards so it picks the variable up.
Paste the key at the prompt only; keep `GEMINI_API_KEY` in the command exactly as written.

macOS / Linux:

```bash
export GEMINI_API_KEY="..."   # add to ~/.zshrc or ~/.bashrc to keep it
```

For Claude models, also `pip install anthropic` and set `ANTHROPIC_API_KEY`.

### 2. Extract one document

```powershell
python docextract.py extract samples/invoices/northwind-office.txt --type invoice
python docextract.py extract my-resume.pdf --type resume --model gemini-3.5-flash-lite
```

Prints the JSON to stdout, and latency / tokens / cost to stderr.

### 3. Benchmark models

```powershell
python docextract.py compare samples
python docextract.py compare samples --models gemini-3.8-flash gemini-3.5-flash-lite
python docextract.py compare samples --effort medium --runs 3
```

| Option | Default | What it does |
| --- | --- | --- |
| `--models` | 3 Gemini models (and 3 Claude models if `ANTHROPIC_API_KEY` is set) | Models to compare |
| `--effort` | `low` | Thinking effort, the same for every model |
| `--runs` | `1` | Repeat each document N times for steadier latency numbers |
| `--parallel` | `1` | Requests in flight per model. Models always run side by side; >1 is faster but latency then includes self-contention and may hit rate limits |
| `--out` | `results` | Where the table (`.md`) and per-document details (`.json`) are saved |

The `.json` report has every extraction and a list of each field a model got wrong.

## Results on the sample set

6 documents (3 invoices, 3 resumes), effort `low`, one request at a time per model,
run on 2026-11-26 with a free-tier key.

| Model | Accuracy | Invoices | Resumes | Latency p50 | Latency p95 | Cost / doc | Cost / 1K docs | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gemini-3-flash-preview` | 97.6% | 96.3% | 98.9% | 2.26s | 2.89s | $0.00095 | $0.95 | 0/6 |
| `gemini-3.5-flash-lite` | 98.2% | 96.3% | 100.0% | 2.47s | 3.12s | $0.00102 | $1.02 | 0/6 |
| `gemini-4-hydrogen` | _pending — free-tier daily quota_ | | | | | | | |

Every miss was a genuine extraction error, not a scoring artefact:

- **Both models computed unit prices** on the catering invoice ($75, $32) by dividing the
  line totals. Those prices aren't printed, and the prompt says not to compute values.
- **"ATTN: Priya Raman" misplaced** — `gemini-3-flash-preview` used it as the customer's
  address, `gemini-3.5-flash-lite` appended it to the customer's name.
- **Inferred location** — `gemini-3-flash-preview` gave a candidate's location as "Lisbon"
  because they work at a Lisbon office; the resume never states where they live.

Six documents is a small sample: treat differences of a point or two as noise, and add
your own documents before choosing a model.

## How the numbers are computed

- **Accuracy** — every field in the ground truth is one point; line items, jobs and education
  entries count per field. Numbers match within a cent, strings ignore case and punctuation,
  long text (addresses, descriptions) tolerates small wording differences, phone numbers
  compare digits. List items are matched regardless of order, and each extra item a model
  invents costs as many points as a real one is worth — so over-extracting doesn't pay.
- **Latency** — wall-clock time of the successful request, p50 and p95. Time spent waiting to
  retry a rate-limited or overloaded request isn't counted.
- **Cost** — from each response's token usage at paid-tier list prices (thinking tokens are
  billed as output). Free-tier keys aren't actually charged; the column is there so models
  can be compared. Prices are in the `MODELS` table at the top of `docextract.py` — update
  them if your provider changes pricing.
- **Errors** — requests that failed. Network and server errors (timeouts, 429, 503) are
  retried first; if they still fail they're listed here but left out of accuracy and cost,
  because there was no answer to score. Invalid JSON or a refusal *is* scored as 0.

## Add your own test documents

Put a document in a folder named `invoices/` or `resumes/` and its ground truth next to it as
`<same name>.expected.json`, shaped like the schema (see the samples). `compare` picks up
every pair it finds.

## Schemas

Defined as Pydantic models in `docextract.py`:

- **Invoice** — number, issue/due dates, currency, vendor (name, address, tax ID), bill-to,
  line items (description, quantity, unit price, amount), subtotal, tax, shipping, total,
  payment terms.
- **Resume** — name, email, phone, location, skills, experience (company, title, start/end,
  location), education (institution, degree, field, graduation year).

Fields not printed in the document come back as `null`; the prompt tells the model not to
compute or guess values.

## Files

| File | Purpose |
| --- | --- |
| `docextract.py` | CLI, schemas, Gemini and Claude backends, benchmark runner |
| `scoring.py` | Field-level accuracy scoring |
| `samples/` | 3 invoices + 3 resumes (fictional) with ground truth |
