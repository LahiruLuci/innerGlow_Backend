
from rag.retriever import retrieve_knowledge

test_messages = [
    "I am scared of speaking in front of my classmates.",
    "I keep thinking everyone will judge me.",
    "How can I calm down when I feel anxious?",
    "I feel lonely and need someone to talk to."
]

for message in test_messages:
    print("\n" + "=" * 60)
    print("USER MESSAGE:", message)
    print("=" * 60)

    results = retrieve_knowledge(message, top_k=3)

    for index, item in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Topic:", item["topic"])
        print("Source:", item["source"])
        print("Distance:", round(item["distance"], 4))
        print("Knowledge:", item["text"])
