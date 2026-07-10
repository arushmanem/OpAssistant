"""Sanity check the eval dataset before running any metrics."""
import json
import re
from document_loader import load_pdf


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# Load resume text
resume_text = normalize(load_pdf("ArushManem_Resume_Updated copy.docx.pdf"))

# Load eval dataset
with open("eval_dataset.json") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} questions from eval_dataset.json\n")

# Count by type
type_counts = {}
for item in dataset:
    type_counts[item["type"]] = type_counts.get(item["type"], 0) + 1
print("Questions by type:")
for t, count in sorted(type_counts.items()):
    print(f"  {t}: {count}")
print()

# Verify every expected substring exists in the resume
print("Verifying substrings exist in resume text:")
all_good = True
for i, item in enumerate(dataset):
    substrings = item["expected_chunk_substrings"]
    
    if substrings is None:
        # Adversarial, skip
        continue
    
    for s in substrings:
        s_norm = normalize(s)
        if s_norm.lower() in resume_text.lower():
            status = "[OK]"
        else:
            status = "[MISS]"
            all_good = False
        print(f"  Q{i+1} {status} {s!r}")

print()
if all_good:
    print("All substrings verified — eval dataset is clean.")
else:
    print("Some substrings don't exist in the resume. Fix these before proceeding.")