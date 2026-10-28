from rag.retriever import retrieve_knowledge

# Five test questions for InnerGlow's CBT knowledge base
test_queries = [
    "How can I manage exam stress?",
    "What can I do when I feel anxious in social situations?",
    "How can I challenge negative thoughts?",
    "How can I improve my mental wellbeing?",
    "How can I maintain progress after overcoming anxiety?"
]


def evaluate_rag():
    for number, query in enumerate(test_queries, start=1):
        print(f"\n{'=' * 60}")
        print(f"QUERY {number}: {query}")
        print("=" * 60)

        results = retrieve_knowledge(query, top_k=3)

        if not results:
            print("No documents retrieved.")
            continue

        for rank, result in enumerate(results, start=1):
            print(f"\nResult {rank}")
            print(f"Source: {result.get('source', 'Unknown')}")
            print(f"Topic: {result.get('topic', 'General')}")
            print(f"Distance: {result.get('distance', 'N/A')}")
            print(f"Text: {result.get('text', '')[:300]}")


if __name__ == "__main__":
    evaluate_rag()