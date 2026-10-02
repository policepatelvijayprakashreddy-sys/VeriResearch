import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables

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

# ============================================
# VERIFICATION AGENT (NLI)
# ============================================

# Set to False to bypass verification (ablation / fast runs)
ENABLE_VERIFICATION = os.getenv("ENABLE_VERIFICATION", "true").lower() == "true"

# HuggingFace NLI model — downloads ~185 MB on first run, cached locally
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# Minimum confidence to trust an NLI label (0.0-1.0)
NLI_CONFIDENCE_THRESHOLD = 0.70

# ============================================
# SOURCE DIVERSITY TARGETS
# ============================================

# Target: 30-50% academic for scientific topics
SOURCE_DIVERSITY_TARGET_MIN = 0.30
SOURCE_DIVERSITY_TARGET_MAX = 0.50

# ============================================
# SEARCH CACHE
# ============================================

SEARCH_CACHE_ENABLED = os.getenv("SEARCH_CACHE_ENABLED", "true").lower() == "true"
SEARCH_CACHE_TTL_HOURS = int(os.getenv("SEARCH_CACHE_TTL_HOURS", "24"))
SEARCH_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "data", "search_cache"
)

# ============================================
# SOURCE RECENCY
# ============================================

# Warn in report if source is older than this many years
SOURCE_RECENCY_WARNING_YEARS = 5

# ============================================
# SCHOLARLY IDENTITY & SOURCE QUALITY VERIFICATION
# ============================================

OPENALEX_BASE_URL = os.getenv("OPENALEX_BASE_URL", "https://api.openalex.org")
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "research-agent@example.com")
OPENALEX_TIMEOUT_SECONDS = int(os.getenv("OPENALEX_TIMEOUT_SECONDS", "6"))

CROSSREF_VERIFY_BASE_URL = os.getenv("CROSSREF_VERIFY_BASE_URL", "https://api.crossref.org/works")
CROSSREF_VERIFY_TIMEOUT_SECONDS = int(os.getenv("CROSSREF_VERIFY_TIMEOUT_SECONDS", "4"))
CROSSREF_VERIFY_MAILTO = os.getenv("CROSSREF_MAILTO") or os.getenv("OPENALEX_MAILTO") or "research-agent@example.com"

QUALITY_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "data", "quality_cache"
)
QUALITY_CACHE_TTL_HOURS = int(os.getenv("QUALITY_CACHE_TTL_HOURS", "168"))  # 7 days

