from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# ---------------- Env ----------------
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY")
# make sure downstream libs see them
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
os.environ["OPENAI_API_KEY"]   = OPENAI_API_KEY or ""

# -------------- RAG wiring --------------
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),  # NOTE: if you want docs in prompt, add "\n\n{context}"
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("chat.html")

# silence favicon 404s in logs (optional)
@app.route("/favicon.ico")
def favicon():
    return ("", 204)

@app.route("/health")
def health():
    return jsonify(status="ok", index=index_name)

@app.route("/get", methods=["GET", "POST"])
def chat():
    """
    Accepts:
      - POST JSON: {"msg": "..."}
      - POST Form: msg=...
      - GET  Query: ?msg=...
    Returns plain text so your front-end shows the string directly.
    """
    msg = None
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        msg = payload.get("msg")
    elif request.method == "POST":
        msg = request.form.get("msg")
    else:
        msg = request.args.get("msg")

    if not msg:
        return "No 'msg' provided", 400

    # be robust across LC versions: provide both keys
    result = rag_chain.invoke({"input": msg, "question": msg})
    answer = result.get("answer", str(result))
    return str(answer)

# --------------- Main -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
