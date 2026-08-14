import os

import chromadb

from typing import TypedDict
from sentence_transformers import SentenceTransformer

from langgraph.graph import StateGraph, END

MOCK_LLM = os.getenv("MOCK_LLM", "1")

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="zepto_docs"
)


class GraphState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list
    confidence: float


KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]


def classify_intent(state):

    query = state["query"].lower()

    intent = "general_question"

    for keyword in KEYWORDS:
        if keyword in query:
            intent = "policy_question"
            break

    state["intent"] = intent

    return state


def retrieve_and_answer(state):

    query = state["query"]

    embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )

    docs = results["documents"][0]
    ids = results["ids"][0]

    top_doc = docs[0]

    if MOCK_LLM == "1":

        answer = (
            "Based on the retrieved context: "
            + top_doc[:200]
        )

    else:

        answer = (
            "Grounded Answer: "
            + top_doc[:300]
        )

    state["answer"] = answer
    state["sources"] = ids
    state["confidence"] = 1.0

    return state


def direct_answer(state):

    if MOCK_LLM == "1":

        answer = (
            "I can only answer questions about "
            "Zepto policies right now."
        )

    else:

        answer = (
            "General answer mode enabled."
        )

    state["answer"] = answer
    state["sources"] = []
    state["confidence"] = 1.0

    return state


def router(state):

    return state["intent"]


builder = StateGraph(GraphState)

builder.add_node(
    "classify_intent",
    classify_intent
)

builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

builder.add_node(
    "direct_answer",
    direct_answer
)

builder.set_entry_point(
    "classify_intent"
)

builder.add_conditional_edges(
    "classify_intent",
    router,
    {
        "policy_question":
        "retrieve_and_answer",

        "general_question":
        "direct_answer"
    }
)

builder.add_edge(
    "retrieve_and_answer",
    END
)

builder.add_edge(
    "direct_answer",
    END
)

graph = builder.compile()