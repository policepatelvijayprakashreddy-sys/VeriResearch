import os

# ============================================
# LLM CONFIGURATION
# ============================================

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

# ============================================
# SEARCH CONFIGURATION
# ============================================

DEFAULT_MAX_RESULTS = 5
MAX_SEARCH_RETRIES = 3
SEARCH_RETRY_DELAY_SECONDS = 1

RESEARCH_INITIAL_MAX_RESULTS = 5
RESEARCH_FOLLOWUP_MAX_RESULTS = 5

LITERATURE_SEARCH_MAX_RESULTS = 8
MAX_ACADEMIC_SOURCES = 8
MAX_WEB_SOURCES = 12

# ============================================
# GRAPH CONTROL
# ============================================

MAX_RESEARCH_ROUNDS = 2

# ============================================
# SEARXNG - FALLBACK GENERAL WEB SEARCH (no API key)
# ============================================

SEARXNG_INSTANCES = [
	"https://searx.be",
	"https://priv.au",
	"https://search.inetol.net",
]

SEARXNG_TIMEOUT_SECONDS = 8

# ============================================
# SEMANTIC SCHOLAR - PRIMARY ACADEMIC SOURCE (no API key required)
# ============================================

SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
SEMANTIC_SCHOLAR_MAX_RESULTS = 8
SEMANTIC_SCHOLAR_TIMEOUT_SECONDS = 10

# ============================================
# CROSSREF - FALLBACK ACADEMIC SOURCE (no API key)
# ============================================

CROSSREF_BASE_URL = "https://api.crossref.org/works"
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "")
CROSSREF_MAX_RESULTS = 8
CROSSREF_TIMEOUT_SECONDS = 10

# ============================================
# ACADEMIC DOMAINS
# ============================================

ACADEMIC_DOMAINS = [
	"pubmed.ncbi.nlm.nih.gov",
	"pmc.ncbi.nlm.nih.gov",
	"sciencedirect.com",
	"springer.com",
	"link.springer.com",
	"wiley.com",
	"onlinelibrary.wiley.com",
	"bmj.com",
	"nature.com",
	"ieee.org",
	"acm.org",
	"biomedcentral.com",
	"arxiv.org",
	"semanticscholar.org",
	"jamanetwork.com",
	"thelancet.com",
	"frontiersin.org",
	"mdpi.com",
	"jstor.org",
	"cell.com",
	"plos.org",
	"tandfonline.com",
	"cambridge.org",
	"oup.com",
	"academic.oup.com",
]
