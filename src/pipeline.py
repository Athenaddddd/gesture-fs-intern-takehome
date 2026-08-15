"""
Document Q&A Pipeline — YOUR WORK GOES HERE.

The knowledge base (loading, chunking, vector store) is already built
for you in knowledge_base.py. Your job is to:

  1. Retrieve relevant chunks and generate an answer
  2. Wire it up into an interactive CLI

Useful docs:
  - Vector store search: https://python.langchain.com/docs/how_to/vectorstores/
  - HuggingFace pipelines: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
"""

import argparse
import os

# 模型缓存统一放在项目目录下（.cache/huggingface），默认从这里读取。
# 默认强制离线（只读本地缓存、不联网）；如需重新下载，请覆盖为 0 并设置镜像。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_ROOT, ".cache", "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# pytest 场景下，测试文件会先于本模块 import transformers，此时 huggingface_hub
# 的缓存路径常量已被固化为用户默认目录。这里显式重定向到项目目录，保证 CLI 与测试一致。
import huggingface_hub.constants as _hf_constants
_hf_constants.HF_HUB_CACHE = os.path.join(_PROJECT_ROOT, ".cache", "huggingface", "hub")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.knowledge_base import build_knowledge_base


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm():
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1: Implement ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_question(vector_store, llm, question: str) -> dict:
    """Retrieve relevant chunks and generate an answer.

    Steps:
      1. Use vector_store.similarity_search(question, k=3) to get
         the top 3 most relevant document chunks.
      2. Combine the chunk text into a single context string.
         (Hint: each chunk has a .page_content attribute)
      3. Format the PROMPT_TEMPLATE with the context and question.
      4. Pass the formatted prompt to llm(...) and extract the
         generated text from the result.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string

    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved
    """
    docs = vector_store.similarity_search(question, k=3)
    sources = [doc.page_content for doc in docs]
    context = "\n\n".join(sources)

    formatted_prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    result = llm(formatted_prompt)
    answer = result[0]["generated_text"]

    return {"answer": answer, "sources": sources}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2: Complete the interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _print_result(result: dict) -> None:
    """Print the retrieved sources and generated answer."""
    print("\n📄 Sources:")
    for i, source in enumerate(result["sources"], start=1):
        preview = source.replace("\n", " ").strip()
        print(f"  {i}. {preview}")
    print(f"\n💬 Answer: {result['answer']}\n")


def _has_documents(data_dir: str) -> bool:
    """Return True if data_dir contains at least one .txt file."""
    if not os.path.isdir(data_dir):
        return False
    for _, _, filenames in os.walk(data_dir):
        if any(name.endswith(".txt") for name in filenames):
            return True
    return False


def main() -> None:
    """Interactive Q&A loop, or one-shot mode via --query.

    Steps:
      1. Build the knowledge base using build_knowledge_base()
         with the data/ directory path.
      2. Load the LLM using get_llm().
      3. Start a loop that:
         - Prompts the user for a question with input()
         - Exits if they type "quit"
         - Calls ask_question() with their input
         - Prints the retrieved sources and the answer
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    parser = argparse.ArgumentParser(
        description="Ask questions about a marketing agency's services, pricing, and process."
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Answer a single question and exit (non-interactive mode).",
    )
    args = parser.parse_args()

    if not _has_documents(data_dir):
        print(f"Error: no .txt documents found under {data_dir}.")
        return

    vector_store = build_knowledge_base(data_dir)
    llm = get_llm()

    if args.query:
        _print_result(ask_question(vector_store, llm, args.query))
        return

    print("Ask a question about our services, pricing, or process.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() == "quit":
            break

        if not question:
            continue

        _print_result(ask_question(vector_store, llm, question))


if __name__ == "__main__":
    main()