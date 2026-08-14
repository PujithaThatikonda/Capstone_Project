from fastapi import FastAPI

from support_assistant.models import QueryRequest,QueryResponse

from support_assistant.graph import graph

app = FastAPI(
    title="Zepto Support Assistant"
)


@app.post(
    "/ask",
    response_model=QueryResponse
)
def ask_question(request: QueryRequest):

    result = graph.invoke(
        {
            "query": request.query
        }
    )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )


@app.get("/")
def home():

    return {
        "message":
        "Zepto Support Assistant Running"
    }