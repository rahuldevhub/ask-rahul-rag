"""Small retrieval check for the supplied resume; not a full RAG evaluation."""

from .retrieval import DEFAULT_INDEX, MiniLM, load_index, search

CASES = [
    ("What finance and invoicing features did Rahul build?", "LedgerFlow"),
    ("What did Rahul build for author records and royalty calculations?", "Co-founder"),
    ("What experience does Rahul have building Snapchat campaigns?", "AR Designer"),
    ("Which college did Rahul attend and what degree did he earn?", "EDUCATION"),
    ("What did Rahul do on Fiverr?", "Freelance UI/UX"),
    ("How did the publishing enquiry app prevent duplicate submissions?", "Publishing Plan"),
    ("Which project uses MiniLM and FAISS for document question answering?", "Resume Intelligence"),
]


def main():
    index, manifest = load_index(DEFAULT_INDEX)
    encoder = MiniLM(manifest["model_revision"])
    passed = 0
    for question, expected in CASES:
        hits = search(index, manifest, encoder, question, top_k=3)
        rank = next((i for i, hit in enumerate(hits, 1)
                     if expected.casefold() in hit["label"].casefold()), None)
        passed += rank is not None
        print(f"{'PASS' if rank else 'FAIL'} | rank {rank or '-'} | {question}")
    print(f"\nExpected evidence in top 3: {passed}/{len(CASES)}")
    print("This checks source retrieval only, not generated answers or factual accuracy of the resume.")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
