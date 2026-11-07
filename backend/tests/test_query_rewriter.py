from backend.app.retrieval.query_rewriter import QueryRewriter


def test_multi_query_prompt_requests_a_keyword_focused_technical_query():
    class FakeLLM:
        prompt = None

        def generate(self, prompt):
            self.prompt = prompt
            return (
                "Which components make up this model?\n"
                "model architecture technical components\n"
                "What are the main elements of the model?",
                "fake-model",
                "api",
            )

    llm = FakeLLM()
    rewriter = QueryRewriter(llm)

    queries = rewriter.generate_multi_queries("What are the two main parts of the model?")

    assert len(queries) == 3
    assert "keyword-focused" in llm.prompt
    assert "canonical technical terms" in llm.prompt
    assert "Do NOT answer the question" in llm.prompt


def test_multi_query_parser_ignores_model_preamble_and_keeps_all_real_variants():
    class FakeLLM:
        def generate(self, prompt):
            return (
                "Here are three alternative search queries:\n"
                "1. Transformer architecture structure\n"
                "2. Transformer encoder and decoder components\n"
                "3. How the Transformer model is organized",
                "fake-model",
                "api",
            )

    queries = QueryRewriter(FakeLLM()).generate_multi_queries(
        "According to the paper, what are the two main parts of the model?"
    )

    assert queries == [
        "Transformer architecture structure",
        "Transformer encoder and decoder components",
        "How the Transformer model is organized",
    ]
