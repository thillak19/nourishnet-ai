"""
NourishNet AI - RAG Customer Support Pipeline
Builds a FAISS vector store from support documents.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models" / "saved" / "rag"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DOCUMENTS = [
    "Our delivery time is typically between 30 to 45 minutes depending on your location and traffic conditions.",
    "You can track your order in real-time using the NourishNet app after placing your order.",
    "We accept payments via UPI, credit card, debit card, net banking, and cash on delivery.",
    "To cancel an order, go to My Orders and click Cancel within 5 minutes of placing the order.",
    "If your food arrives cold or damaged, contact support within 30 minutes for a full refund.",
    "Our restaurants operate from 8 AM to 11 PM. Delivery is available during these hours only.",
    "Minimum order value is Rs 99. Free delivery is available on orders above Rs 299.",
    "Refunds are processed within 5 to 7 business days to your original payment method.",
    "Our customer support is available 24/7 via chat, email, and phone at 1800-NOURISH.",
    "Premium members get free delivery on all orders, priority support, and exclusive discounts.",
    "To become a premium member, go to Profile and select Upgrade to Premium for Rs 99 per month.",
    "You can save multiple delivery addresses in your profile for faster checkout.",
    "All our delivery partners are verified, trained, and tracked in real time for your safety.",
    "You can report a missing item directly from the order details page within 24 hours.",
    "NourishNet offers a loyalty program where every Rs 100 spent earns 10 NourishPoints.",
]


def build_vector_store():
    docs = [Document(page_content=text) for text in DOCUMENTS]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    output_path = MODEL_DIR / "nourishnet_vectorstore"
    vectorstore.save_local(str(output_path))
    return vectorstore


def main():
    print("Building RAG vector store...")
    vectorstore = build_vector_store()
    print(f"Vector store saved to {MODEL_DIR}")

    query = "How do I cancel my order?"
    results = vectorstore.similarity_search(query, k=2)
    print(f"\nTest query: {query}")
    for i, result in enumerate(results, 1):
        print(f"Result {i}: {result.page_content}")


if __name__ == "__main__":
    main()