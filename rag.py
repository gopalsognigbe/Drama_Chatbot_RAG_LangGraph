import json
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START

# =========================
# 1. LOAD JSON
# =========================
with open("data/voirdrama.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# =========================
# 2. DOCUMENTS
# =========================
documents = []

for item in data:
    content = f"""
Titre: {item.get('title', '')}
Synopsis: {item.get('synopsis', '')}
Genre: {", ".join(item.get('genre', []))}
Pays: {item.get('country', '')}
Statut: {item.get('statut', '')}
Episodes: {item.get('episode', '')}
Note: {item.get('note', '')}
"""

    metadata = {
        "title": item.get("title", ""),
        "country": item.get("country", ""),
        "url": item.get("url", "")
    }

    if item.get("genre"):
        metadata["genre"] = item["genre"]

    documents.append(Document(page_content=content, metadata=metadata))

# =========================
# 3. EMBEDDINGS
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# 4. VECTOR STORE (persistant)
# =========================
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# =========================
# 5. RETRIEVER
# =========================
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10}
)

# =========================
# 6. LLM
# =========================
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="conversational"
    )
)

# =========================
# 7. PROMPT
# =========================
prompt = ChatPromptTemplate.from_messages([
    ("system", """
Tu es un expert en dramas.

Utilise uniquement les données fournies.
Réponds sous forme de liste.
"""),
    ("human", "Question: {question}\n\nContexte:\n{context}")
])

# =========================
# 8. LANGGRAPH
# =========================
class State(TypedDict):
    input: str
    context: list
    answer: str

def retrieve(state: State):
    docs = retriever.invoke(state["input"])
    return {"context": docs}

def generate(state: State):
    context_text = "\n\n".join(doc.page_content for doc in state["context"])

    response = llm.invoke(
        prompt.format(
            question=state["input"],
            context=context_text
        )
    )

    return {"answer": response.content}

graph = StateGraph(State)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")

rag_chain = graph.compile()