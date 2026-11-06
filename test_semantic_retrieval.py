from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import time

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

claim = "Deep learning improves image classification accuracy."

evidence_samples = {
    "Paraphrase": (
        "Neural network architectures achieve higher "
        "recognition rates."
    ),
    "Contradiction": (
        "Deep learning does not improve image classification accuracy."
    ),
    "Related but neutral": (
        "Image classification datasets contain thousands "
        "of labeled images."
    ),
    "Unrelated": (
        "Ancient Roman aqueducts used sophisticated masonry "
        "and hydraulic mortar construction."
    ),
}

print("Loading model...")
try:
    model = SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
except Exception:
    model = SentenceTransformer(MODEL_NAME, device="cpu")

sentences = [claim] + list(evidence_samples.values())
start = time.perf_counter()
embeddings = model.encode(
    sentences,
    normalize_embeddings=True,
    convert_to_tensor=True,
)
elapsed = time.perf_counter() - start
print(f"Encoding time: {elapsed:.2f}s\n")

claim_embedding = embeddings[0]
evidence_embeddings = embeddings[1:]
similarities = util.cos_sim(claim_embedding, evidence_embeddings)[0]

print(f"Claim: '{claim}'\n")
print(f"{'Category':<22} | {'Cosine Sim':<10} | Evidence Text")
print("-" * 85)
for (category, text), sim in zip(evidence_samples.items(), similarities):
    print(f"{category:<22} | {sim.item():<10.4f} | {text}")
