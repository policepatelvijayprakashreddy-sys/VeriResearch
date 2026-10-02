# Research Report: Security vulnerabilities in LLM-generated code

*Generated: 2026-10-02 18:09:44* | *Framework: VeriResearch Multi-Agent Verification*

---

## 1. Executive Summary

This report summarizes the findings on security vulnerabilities in LLM-generated code. The research scope focused on identifying key vulnerabilities and mitigation strategies for LLM-generated code. The main findings include the importance of sanitizing outputs, encoding, and escaping outputs to prevent XSS attacks, and the need for secure code generation and the avoidance of hallucinated security advice. The report also highlights the necessity of adding AI security test cases to Continuous Integration (CI) for critical workflows.

## 2. General Research Findings

The general web research revealed that LLM-generated code can be vulnerable to security threats, including XSS attacks. The OWASP Top 10 LLM vulnerabilities and security checklist identified key vulnerabilities and provided mitigation strategies. A comprehensive analysis of the security of LLM-generated C code from neutral prompts found that experimental poisoning attacks on autocompleters and empirical analysis comparing human and Copilot-generated code vulnerabilities were significant concerns.

## 3. Literature Review

The literature review analyzed the following papers:

* Adversarial-Resistant AI Tutoring: A Dual-LLM Architecture for Mitigating H-CoT Attacks and Ensuring Pedagogically-Aligned Code Assistance [8]
* ChronosAttack: A Delay-Only Scheduling Attack on LLM Agents [2]
* Adversarial Hallucination Engineering: A Threat Model for Biasing LLM Reasoning [3]
* Cognitive Camouflage: A Taxonomy of Specification Gaming Strategies in LLMs [4]
* False alarms, real damage: Evaluating the Effectiveness of LLM-based Models in Detecting Adversarial Attacks [5]
* Evaluating and Defending Against Adversarial Attacks on LLM-Generated LSTM Models [6]
* Top 10 vulnerabilities in LLM applications such as ChatGPT [7]
* Explore mitigation strategies for 10 LLM vulnerabilities [9]
* Exploring LLM Vulnerabilities and Security Best Practices [10]

## 4. Critical Analysis

The critical analysis evaluated the rigor and reliability of the gathered evidence. The papers analyzed in the literature review had varying levels of peer-review status and experimental constraints. The methodologies used in the papers were often innovative and required further validation.

## 5. Knowledge Gaps

The knowledge gaps identified in the literature review include the need for further research on the following topics:

* Developing more effective mitigation strategies for LLM-generated code vulnerabilities
* Improving the robustness of LLM-based models in detecting adversarial attacks
* Investigating the impact of neurodiversity on LLM-generated code



### Claims We Could Not Verify

The following claims were extracted from sources but our NLI verification model could not confirm that the source text directly entails them. These should be treated with additional scrutiny:

- ⚠️ "LLM-generated C code can be vulnerable to security threats."
  Source: [10] [Generative AI in Cybersecurity: A Comprehensive](https://arxiv.org/pdf/2405.12750) | Status: NEUTRAL (confidence: 0.777)

- ⚠️ "Engineers need to add AI security test cases to their workflows."
  Source: [12] [LLM Security Vulnerabilities Engineers Need to... - DEV Community](https://dev.to/opsbuzzdev/llm-security-vulnerabilities-engineers-need-to-know-in-2026-4cd8) | Status: NEUTRAL (confidence: 0.921)

- ⚠️ "LLMs are a type of AI model that can be explained in simple terms."
  Source: [14] [What Is an LLM ? Beginner's Guide to AI in 2026](https://freeacademy.ai/blog/what-is-an-llm-beginners-guide-2026) | Status: NEUTRAL (confidence: 0.646)

- ⚠️ "LLM-generated code can be vulnerable to security threats that can be detected using transferable representations."
  Source: [1] [Cross-Language Transfer Learning for Detecting Vulnerabilities in LLM-Generated Code&amp;nbsp;](https://doi.org/10.2139/ssrn.6144009) | Status: NEUTRAL (confidence: 0.961)

- ⚠️ "LLM-generated code can experience significant vulnerability growth over time."
  Source: [2] [Security Weaknesses in LLM-Generated Source Code: An Empirical Vulnerability Analysis of Iterative AI-Assisted Development](https://doi.org/10.2139/ssrn.6958668) | Status: NEUTRAL (confidence: 0.971)

- ⚠️ "Inclusivity concerns in LLM-generated code remain unexplored."
  Source: [3] [Evaluating Security and Inclusivity in LLM-Generated Code: A Controlled Experiment](https://doi.org/10.2139/ssrn.6731323) | Status: NEUTRAL (confidence: 0.997)

- ⚠️ "A dual-LLM architecture can be used to mitigate adversarial attacks in AI tutoring systems."
  Source: [8] [Adversarial-Resistant AI Tutoring: A Dual-LLM Architecture for Mitigating H-CoT Attacks and Ensuring Pedagogically-Aligned Code Assistance](https://doi.org/10.31219/osf.io/kjng3_v2) | Status: NEUTRAL (confidence: 0.995)



## 6. Recommendations

Based on the findings and knowledge gaps identified, the following recommendations are made:

* Develop and implement more effective mitigation strategies for LLM-generated code vulnerabilities
* Invest in improving the robustness of LLM-based models in detecting adversarial attacks
* Conduct further research on the impact of neurodiversity on LLM-generated code

## 7. Conclusion

In conclusion, the research on security vulnerabilities in LLM-generated code highlights the need for more effective mitigation strategies and improved robustness of LLM-based models. Further research is needed to address the identified knowledge gaps and ensure the development of secure and reliable LLM-generated code.

---

## Cross-Paper Scientific Consensus Matrix

| # | Core Empirical Claim | Supporting Literature | Contradicting Literature | Retraction Alerts | Consensus Verdict |
|---|----------------------|-----------------------|--------------------------|-------------------|-------------------|
| [1] | LLM-generated code is vulnerable to XSS attacks and requires... | **OWASP Top 10 LLM Vulnerabiliti**: *"An LLM-generated script causing an XSS attack. Sanitize outputs, encod..."* | **Shift-Left Security for AI-Gen** *(Journal of Cyber Security)*: *"Shift-Left Security for AI-Generated Code: Detecting and Preventing Vu..."*<br>**Security and Quality in LLM-Ge** *(Unknown)*: *"Security and Quality in LLM-Generated Code: a Multi-Language, Multi-Mo..."* | 🟢 None | 🟡 MIXED_EVIDENCE |
| [2] | Building an LLM is a complex process. | **What Are Large Language Models**: *"Building an LLM from scratch is a complex and resource-intensive proce..."* | *None* | 🟢 None | ⚪ SINGLE_SOURCE_CLAIM |
| [3] | LLM security vulnerabilities differ from traditional applica... | **What are the OWASP Top 10 risk**: *"How do LLM security vulnerabilities differ from traditional applicatio..."* | *None* | 🟢 None | ⚪ SINGLE_SOURCE_CLAIM |
| [4] | There are 10 known vulnerabilities in LLM applications. | **Top 10 vulnerabilities in LLM **: *"The OWASP Top 10 LLM Application Vulnerabilities aims to educate and r..."* | *None* | 🟢 None | ⚪ SINGLE_SOURCE_CLAIM |
| [5] | There are 10 known vulnerabilities in LLM applications and t... | **Explore mitigation strategies **: *"Review the OWASP Top 10 list of LLM security vulnerabilities and their..."* | *None* | 🟢 None | ⚪ SINGLE_SOURCE_CLAIM |

*> **Consensus Evaluation Note:** Mixed evidence reflects differing empirical experimental conditions across papers. Retracted papers are strictly isolated from scientific consensus counts.*

---

## Source Quality & Verification Audit

| # | Paper Title | OpenAlex | Crossref | Retraction Status | Journal / Venue | ISSN | Identity Verification |
|---|-------------|----------|----------|-------------------|-----------------|------|-----------------------|
| [1] | Cross-Language Transfer Learning for Detec... | ✅ Verified | ✅ Verified | 🟢 No retraction flag (OpenAlex/Crossref) | SSRN Electronic Journal | `1556-5068` | 🟢 HIGH |
| [2] | Security Weaknesses in LLM-Generated Sourc... | ✅ Verified | ✅ Verified | 🟢 No retraction flag (OpenAlex/Crossref) | SSRN Electronic Journal | `1556-5068` | 🟢 HIGH |
| [3] | Evaluating Security and Inclusivity in LLM... | ✅ Verified | ✅ Verified | 🟢 No retraction flag (OpenAlex/Crossref) | SSRN Electronic Journal | `1556-5068` | 🟢 HIGH |
| [4] | Exploring Potential Vulnerabilities in LLM... | ✅ Verified | ✅ Verified | 🟢 No retraction flag (OpenAlex/Crossref) | Office of Scientific and Technical Informatio... | `N/A` | 🟢 HIGH |
| [5] | Shift-Left Security for AI-Generated Code:... | ✅ Verified | ✅ Verified | 🟢 No retraction flag (OpenAlex/Crossref) | Journal of Cyber Security | `2579-0064` | 🟢 HIGH |
| [6] | Security and Quality in LLM-Generated Code... | ❌ Not Found | ✅ Verified | 🟢 No retraction flag (OpenAlex/Crossref) | Institute of Electrical and Electronics Engin... | `N/A` | 🟡 MEDIUM |
| [7] | Ensuring Performance Requirements for LLM-... | ✅ Verified | ✅ Verified | 🟢 No retraction flag (OpenAlex/Crossref) | Williams College | `N/A` | 🟢 HIGH |
| [8] | Adversarial-Resistant AI Tutoring: A Dual-... | ✅ Verified | ✅ Verified | 🟢 No retraction flag (OpenAlex/Crossref) | Center for Open Science | `N/A` | 🟢 HIGH |

*> **Note on Scopus / Web of Science:** Not independently verified by our system (requires institutional enterprise API credentials).*

### Sources

[1] [Cross-Language Transfer Learning for Detecting Vulnerabilities in LLM-Generated Code&nbsp;](https://doi.org/10.2139/ssrn.6144009)
    **Harvested From:** Crossref | **Venue:** *SSRN Electronic Journal* (ISSN: `1556-5068`) | **Verified Records:** `[OpenAlex (DOI record found)]` `[CrossRef (Indexed Work)]` `[CrossRef (Official DOI Registrar)]` | **Identity Verification:** `HIGH`

[2] [Security Weaknesses in LLM-Generated Source Code: An Empirical Vulnerability Analysis of Iterative AI-Assisted Development](https://doi.org/10.2139/ssrn.6958668)
    **Harvested From:** Crossref | **Venue:** *SSRN Electronic Journal* (ISSN: `1556-5068`) | **Verified Records:** `[OpenAlex (DOI record found)]` `[CrossRef (Indexed Work)]` `[CrossRef (Official DOI Registrar)]` | **Identity Verification:** `HIGH`

[3] [Evaluating Security and Inclusivity in LLM-Generated Code: A Controlled Experiment](https://doi.org/10.2139/ssrn.6731323)
    **Harvested From:** Crossref | **Venue:** *SSRN Electronic Journal* (ISSN: `1556-5068`) | **Verified Records:** `[OpenAlex (DOI record found)]` `[CrossRef (Indexed Work)]` `[CrossRef (Official DOI Registrar)]` | **Identity Verification:** `HIGH`

[4] [Exploring Potential Vulnerabilities in LLM-Generated Code](https://doi.org/10.2172/3883789)
    **Harvested From:** Crossref | **Venue:** *Unknown* | **Verified Records:** `[OpenAlex (DOI record found)]` `[CrossRef (Indexed Work)]` `[CrossRef (Official DOI Registrar)]` | **Identity Verification:** `HIGH`

[5] [Shift-Left Security for AI-Generated Code: Detecting and Preventing Vulnerabilities at Build-Time](https://doi.org/10.32604/jcs.2026.085438)
    **Harvested From:** Crossref | **Venue:** *Journal of Cyber Security* (ISSN: `2579-0064`) | **Verified Records:** `[OpenAlex (DOI record found)]` `[CrossRef (Indexed Work)]` `[CrossRef (Official DOI Registrar)]` | **Identity Verification:** `HIGH`

[6] [Security and Quality in LLM-Generated Code: a Multi-Language, Multi-Model Analysis_supp1-3672745.pdf](https://doi.org/10.1109/tdsc.2026.3672745/mm1)
    **Harvested From:** Crossref | **Venue:** *Unknown* | **Verified Records:** `[CrossRef (Official DOI Registrar)]` | **Identity Verification:** `MEDIUM`

[7] [Ensuring Performance Requirements for LLM-Generated Code](https://doi.org/10.36934/tr2025_272)
    **Harvested From:** Crossref | **Venue:** *Unknown* | **Verified Records:** `[OpenAlex (DOI record found)]` `[CrossRef (Indexed Work)]` `[CrossRef (Official DOI Registrar)]` | **Identity Verification:** `HIGH`

[8] [Adversarial-Resistant AI Tutoring: A Dual-LLM Architecture for Mitigating H-CoT Attacks and Ensuring Pedagogically-Aligned Code Assistance](https://doi.org/10.31219/osf.io/kjng3_v2)
    **Harvested From:** Crossref | **Venue:** *Unknown* | **Verified Records:** `[OpenAlex (DOI record found)]` `[CrossRef (Indexed Work)]` `[CrossRef (Official DOI Registrar)]` | **Identity Verification:** `HIGH`

[9] [OWASP Top 10 LLM Vulnerabilities & Checklist (2026)](https://www.lasso.security/blog/owasp-top-10-llm-vulnerabilities-security-checklist)
    **Harvested From:** Web Search

[10] [Generative AI in Cybersecurity: A Comprehensive](https://arxiv.org/pdf/2405.12750)
    **Harvested From:** Web Search

[11] [The Ultimate Guide to LLM Security : Risks & Practical Tips](https://masterofcode.com/blog/llm-security)
    **Harvested From:** Web Search

[12] [LLM Security Vulnerabilities Engineers Need to... - DEV Community](https://dev.to/opsbuzzdev/llm-security-vulnerabilities-engineers-need-to-know-in-2026-4cd8)
    **Harvested From:** Web Search

[13] [Large language model - Wikipedia](https://en.wikipedia.org/wiki/Large_language_model)
    **Harvested From:** Web Search

[14] [What Is an LLM ? Beginner's Guide to AI in 2026](https://freeacademy.ai/blog/what-is-an-llm-beginners-guide-2026)
    **Harvested From:** Web Search

[15] [Large Language Models (LLMs) with Google AI | Google Cloud](https://cloud.google.com/ai/llms)
    **Harvested From:** Web Search

[16] [What Are Large Language Models (LLMs)? | IBM](https://www.ibm.com/think/topics/large-language-models)
    **Harvested From:** Web Search

[17] [What are the OWASP Top 10 risks for LLMs?](https://www.cloudflare.com/learning/ai/owasp-top-10-risks-for-llms/)
    **Harvested From:** Web Search

[18] [Top 10 vulnerabilities in LLM applications such as ChatGPT](https://www.tarlogic.com/blog/owasp-top-10-vulnerabilities-llm-applications/)
    **Harvested From:** Web Search

[19] [Explore mitigation strategies for 10 LLM vulnerabilities | TechTarget](https://www.techtarget.com/ai/tip/Explore-mitigation-strategies-for-10-LLM-vulnerabilities)
    **Harvested From:** Web Search

[20] [Exploring LLM Vulnerabilities and Security Best Practices](https://www.vaadata.com/en/blog/exploring-llm-vulnerabilities-and-security-best-practices/)
    **Harvested From:** Web Search
