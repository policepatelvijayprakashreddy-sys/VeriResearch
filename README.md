# VeriResearch 🔬🛡️

> **Autonomous Multi-Agent Verified Academic Research System**  
> Grounded literature discovery, dual-registry provenance auditing, zero-shot NLI claim verification, and cross-paper consensus synthesis.

---

## 🌟 Key Features

1. **Dual-Registry Provenance & Retraction Audit**
   - Cross-references DOIs across **OpenAlex** (including Retraction Watch metadata) and **Crossref**.
   - Accurately tracks journal venues, publication years, publishers, and retraction statuses (`🟢 No retraction flag (OpenAlex/Crossref)`).

2. **Stage-1 Grounding: True DeBERTa NLI Claim Verification**
   - Natural Language Inference using `cross-encoder/nli-deberta-v3-small` running locally on CPU/GPU.
   - Evaluates premise-hypothesis pairs for **ENTAILMENT**, **CONTRADICTION**, and **NEUTRAL** to eliminate hallucinations.

3. **Stage-2 Synthesis: Cross-Paper Consensus Engine**
   - Synthesizes findings across multiple peer-reviewed papers into unified consensus summaries and identifies open research debates.

4. **Multi-Agent Orchestration with LangGraph**
   - Modular pipeline containing Research, Literature, Evidence Extraction, Source Quality, Verification, Consensus, Critic, and Report Generation agents.

5. **Automated Evaluation Harness**
   - Automated quality evaluation scoring Citation Accuracy, Evidence Grounding, Academic Depth, Structural Integrity, and Halting/Hallucination rates on a 0–10 scale.

6. **Rich Terminal Experience**
   - Live progress banners, execution tables, and evaluation panels powered by `rich`.

---

## 🏗️ Architecture

```
                       User Topic Prompt
                              │
                              ▼
                   ┌─────────────────────┐
                   │   Research Agent    │ ── (Web Search / DDG)
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Literature Agent   │ ── (OpenAlex / Crossref / Semantic Scholar)
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │Source Quality Agent │ ── (Dual-Registry Retraction & Venue Audit)
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   Evidence Agent    │ ── (Claim Extraction & Context)
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Verification Agent  │ ── (Local DeBERTa-v3 NLI Model)
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   Consensus Agent   │ ── (Cross-Paper Synthesis & Debate Detection)
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │    Report Agent     │ ── (Grounded Markdown Synthesis)
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Evaluation Harness  │ ── (Automated Scorecards & Metrics)
                   └─────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **[Ollama](https://ollama.com/)** with `llama3.2:3b` (or your preferred local/remote model):
  ```bash
  ollama run llama3.2:3b
  ```

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/policepatelvijayprakashreddy-sys/VeriResearch.git
cd VeriResearch

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Research Pipeline
```bash
# Interactive mode
python main.py

# Specify a topic directly
python main.py --topic "Security vulnerabilities in LLM-generated code"

# Standalone project recommendation mode
python main.py --recommend-only --topic "Federated learning medical imaging privacy"
```

---

## 🧪 Verification Benchmarks

Run the standalone verification test suites to validate NLI and provenance auditing:

```bash
# Cross-Paper Consensus Synthesis Test
python test_consensus_verification.py

# Crossref & OpenAlex Dual-Registry Provenance Test
python test_crossref_verification.py
```

---

## 📊 Sample Evaluation Output

```
================================================================================
                    VERIRESEARCH EVALUATION SCORECARD
================================================================================
  Topic:                 Security vulnerabilities in LLM-generated code
  Overall Score:         7.64 / 10.00
  Verdict:               PASS
--------------------------------------------------------------------------------
  Dimension Scores:
    - Structure & Length:       0.98 / 1.00
    - Citation Count & Ratio:   0.88 / 1.00
    - Grounding & NLI:          0.90 / 1.00
    - Quality Table Complete:   1.00 / 1.00
    - Provenance & Retraction:  0.88 / 1.00
    - Hallucination / Contrad:  0.00%
================================================================================
```

---

## 📄 License
MIT License.
