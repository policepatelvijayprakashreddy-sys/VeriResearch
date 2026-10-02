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

---

## 📊 Live Evaluation Scorecard (`Score: 7.64 / 10`)

The automated evaluation harness grades every run across rule-based factual metrics, grounding density, citation coverage, and LLM-as-judge dimensions:

```
================================================================================
                    VERIRESEARCH EVALUATION SCORECARD
================================================================================
  Topic:                 Security vulnerabilities in LLM-generated code
  Overall Score:         7.64 / 10.00
  Verdict:               PASS (Clean Provenance & Zero Contradictions)
================================================================================
```

### 1. Rule-Based Quantitative Metrics

| Metric | Measured Value | Target / Benchmark | Status |
|:---|:---|:---|:---|
| **Sections Complete** | **7 / 7** | 7 required sections | ✅ 100% |
| **Inline Citations** | **16 citations** | $\ge$ 10 citations | ✅ High Coverage |
| **Citation Coverage** | **100%** | $\ge$ 85% | ✅ Fully Grounded |
| **Academic Literature Sources** | **8 papers** | $\ge$ 5 peer-reviewed | ✅ High Quality |
| **Web Sources** | **12 sources** | $\ge$ 8 trusted domains | ✅ High Diversity |
| **Source Diversity Ratio** | **0.40** | $\ge$ 0.30 | ✅ Multi-Registry |
| **Report Word Count** | **1,738 words** | 1,200 – 2,500 words | ✅ Comprehensive |
| **Evidence Density** | **8.63 items / 1k words** | $\ge$ 6.0 / 1k | ✅ Rich Context |
| **DeBERTa NLI Hallucination Rate** | **0.0%** | < 5.0% | 🛡️ Zero Contradiction |

### 2. LLM-as-Judge Evaluation (1–10 Scale)

| Evaluation Dimension | Score | Assessment |
|:---|:---:|:---|
| **Clarity & Flow** | **8 / 10** | Coherent multi-section synthesis with logical transitions |
| **Critical Analysis** | **8 / 10** | Rigorous evaluation of methodology & threat models |
| **Depth & Technical Detail** | **7 / 10** | Detailed discussion of XSS, AST-based detectors & H-CoT attacks |
| **Factual Grounding** | **6 / 10** | Explicit separation of verified and unverified claims |
| **Recommendation Quality** | **5 / 10** | Specific directions for future empirical verification |
| **Judge Average** | **6.8 / 10** | **Solid Academic Foundation** |

---

## 🔬 Sample Research Report Output & Provenance Audit

Below is the verified source provenance table extracted directly from the generated report:

| # | Paper Title | OpenAlex | Crossref | Retraction Status | Journal / Venue | ISSN | Identity Verification |
|:---|:---|:---:|:---:|:---|:---|:---:|:---:|
| **[1]** | Cross-Language Transfer Learning for Detecting Vulnerabilities in LLM-Generated Code | ✅ Verified | ✅ Verified | 🟢 No retraction flag | SSRN | `1556-5068` | 🟢 HIGH |
| **[2]** | Security Weaknesses in LLM-Generated Source Code: Empirical Vulnerability Analysis | ✅ Verified | ✅ Verified | 🟢 No retraction flag | SSRN | `1556-5068` | 🟢 HIGH |
| **[3]** | Exploring LLM Vulnerabilities and Security Best Practices | ✅ Verified | ✅ Verified | 🟢 No retraction flag | OSF Preprints | `N/A` | 🟡 MEDIUM |
| **[4]** | LLM-Assisted Code Generation: Vulnerabilities, Exploits, and Mitigation Strategies | ✅ Verified | ✅ Verified | 🟢 No retraction flag | OSTI | `N/A` | 🟡 MEDIUM |
| **[5]** | False alarms, real damage: LLM Models in Detecting Adversarial Attacks | ✅ Verified | ✅ Verified | 🟢 No retraction flag | IEEE TDSC | `1545-5971` | 🟢 HIGH |
| **[6]** | Evaluating and Defending Against Adversarial Attacks on LLM-Generated Models | ✅ Verified | ✅ Verified | 🟢 No retraction flag | IEEE Trans. AI | `2691-4581` | 🟢 HIGH |
| **[7]** | Top 10 vulnerabilities in LLM applications | ✅ Verified | ✅ Verified | 🟢 No retraction flag | Computers & Security | `0167-4048` | 🟢 HIGH |
| **[8]** | Adversarial-Resistant AI Tutoring: Dual-LLM Architecture for H-CoT Attacks | ✅ Verified | ✅ Verified | 🟢 No retraction flag | ACM TOCE | `1946-6226` | 🟢 HIGH |

---

## 📁 Sample Research Reports & Evaluations

Explore complete sample outputs generated by VeriResearch:
- 📄 [**Full Research Report (Security vulnerabilities in LLM-generated code)**](reports/20261002_180944_Security_vulnerabilities_in_LLM-generated_code.md)
- 📊 [**Automated Evaluation Scorecard (Score: 7.64 / 10)**](reports/20261002_181025_Security_vulnerabilities_in_LLM-generated_code_eval.md)
- 🚗 [**Full Research Report (AI in Automobile)**](reports/20261002_170207_AI_in_automobile.md)
- 🚁 [**Full Research Report (UAV Localisation)**](reports/20261002_160518_uav_localisation.md)
- 🏥 [**Full Research Report (AI in Healthcare)**](reports/20261001_093432_AI_in_healthcare.md)
- 💡 [**Project Recommendations (Federated learning medical imaging privacy)**](reports/20261002_163029_federated_learning_medical_imaging_recommendations.md)
- 📚 [**Complete Reports Catalog (31 Reports & Evaluations Index)**](reports/README.md)

---

## 📄 License
MIT License.
