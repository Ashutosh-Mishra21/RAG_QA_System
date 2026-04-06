def faithfulness_score(answer, context, llm):

    prompt = f"""
You are evaluating a RAG system.

Check whether the answer is fully supported by the given context.

CONTEXT:
{context}

ANSWER:
{answer}

Reply ONLY with:
YES or NO
"""

    result = llm.generate(prompt)

    return 1 if "YES" in result.upper() else 0


def relevance_score(answer, query, llm):

    prompt = f"""
Does the answer correctly answer the question?

QUESTION:
{query}

ANSWER:
{answer}

Reply ONLY with YES or NO.
"""

    result = llm.generate(prompt)

    return 1 if "YES" in result.upper() else 0
