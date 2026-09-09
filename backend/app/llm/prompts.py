SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer the user's question using ONLY the "
    "context excerpts below, which come from the user's own uploaded documents. "
    "If the context does not fully answer the question, say what is missing instead "
    "of guessing. Never invent facts, filenames, or page numbers that are not in the "
    "context. Answer in the same language as the question."
)


def build_user_prompt(question: str, context_blocks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_blocks)
    return (
        f"Context excerpts:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}\n\n"
        "Answer using only the context above."
    )


def format_context_block(filename: str, page: int, content: str) -> str:
    return f"[Source: {filename}, page {page}]\n{content}"
