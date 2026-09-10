import os
from types import SimpleNamespace

import pytest
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document

from raggy import pipeline
from raggy.pipeline import SCORE_KEY

# A synthetic source path. Built with os.path.join because the filter is only
# ever compared against strings the way `normcase` folds them, which differs by
# platform: on Windows it lower-cases and back-slashes, on POSIX it does neither.
DOC = os.path.join(os.sep, "docs", "a.pdf")


def _message_contents(messages):
    """Flatten (role, content) tuples / BaseMessage objects into their content."""
    contents = []
    for m in messages:
        if isinstance(m, list):
            contents.extend(_message_contents(m))
        elif isinstance(m, tuple):
            contents.append(m[1])
        elif isinstance(m, str):
            contents.append(m)
        else:
            contents.append(m.content)
    return contents


class FakeLLM:
    """Chain-compatible fake chat model: callable (for ``| llm``) and ``.invoke``."""

    def __init__(self, capture, output="final answer"):
        self.capture = capture
        self.output = output

    def __call__(self, prompt_input):
        self.capture["prompt_messages"] = self._normalize(prompt_input)
        return self.output

    def invoke(self, messages):
        self.capture["prompt_messages"] = self._normalize(messages)
        return self.output

    @staticmethod
    def _normalize(prompt_input):
        if hasattr(prompt_input, "to_messages"):
            return prompt_input.to_messages()
        if isinstance(prompt_input, list):
            messages = []
            for m in prompt_input:
                if not isinstance(m, (str, tuple)):
                    messages.extend(FakeLLM._normalize(m))
                else:
                    messages.append(m)
            return messages
        return prompt_input


class FakeModel(BaseCrossEncoder):
    def __init__(self, scores):
        self.scores = scores

    def score(self, pairs):
        return self.scores


def _docs(*contents):
    return [Document(page_content=c) for c in contents]


@pytest.mark.parametrize(
    ("retrieve_k", "alpha", "expected"),
    [
        (50, 0.5, (25, 25)),
        (50, 1.0, (50, 0)),
        (50, 0.0, (0, 50)),
        (10, 0.7, (7, 3)),
        (10, 0.25, (2, 8)),
        # A share small enough to round to zero switches that arm off, the
        # same as asking for the extreme outright.
        (50, 0.001, (0, 50)),
        (50, 0.999, (50, 0)),
    ],
)
def test_split_retrieval_budget(retrieve_k, alpha, expected):
    assert pipeline.split_retrieval_budget(retrieve_k, alpha) == expected


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_split_retrieval_budget_always_sums_to_retrieve_k(alpha):
    assert sum(pipeline.split_retrieval_budget(37, alpha)) == 37


def _retriever_probe(monkeypatch, captured, bm25=None):
    """Stub out everything get_retriever builds, recording the arguments."""

    class FakeVectorstore:
        def as_retriever(self, **kwargs):
            captured["dense"] = kwargs
            return "dense-retriever"

    class FakeEnsemble:
        def __init__(self, **kwargs):
            captured["ensemble"] = kwargs

    def fake_bm25(db_directory, k, sources=None):
        captured["bm25"] = {"db_directory": db_directory, "k": k}
        captured["bm25_sources"] = sources
        if bm25 == "missing":
            raise FileNotFoundError("no index")
        return "bm25-retriever"

    def fake_compressor_cls(**kwargs):
        captured["compressor"] = kwargs
        return "compressor"

    def fake_compression_retriever(**kwargs):
        captured["compression"] = kwargs
        return "compressed-retriever"

    monkeypatch.setattr(pipeline, "EnsembleRetriever", FakeEnsemble)
    monkeypatch.setattr(pipeline, "get_bm25_retriever", fake_bm25)
    monkeypatch.setattr(pipeline, "ScoreAnnotatingReranker", fake_compressor_cls)
    monkeypatch.setattr(
        pipeline, "ContextualCompressionRetriever", fake_compression_retriever
    )
    monkeypatch.setattr(pipeline, "get_cross_encoder", lambda model: f"encoder:{model}")
    return FakeVectorstore()


def test_get_retriever_splits_budget_between_arms(monkeypatch):
    captured = {}
    vectorstore = _retriever_probe(monkeypatch, captured)

    result = pipeline.get_retriever(
        vectorstore,
        retrieve_k=10,
        rerank_model="reranker-model",
        rerank_k=3,
        db_directory="./persist",
        hybrid_alpha=0.7,
    )

    assert result == "compressed-retriever"
    # alpha 0.7 of a 10-chunk budget: 7 dense, 3 lexical.
    assert captured["dense"] == {
        "search_type": "similarity",
        "search_kwargs": {"k": 7},
    }
    assert captured["bm25"] == {"db_directory": "./persist", "k": 3}
    assert captured["ensemble"]["retrievers"] == ["dense-retriever", "bm25-retriever"]
    # Fusion weights stay uniform: alpha already spent its influence on the split.
    assert captured["ensemble"]["weights"] == [0.5, 0.5]


def test_get_retriever_always_wraps_the_cross_encoder(monkeypatch):
    captured = {}
    vectorstore = _retriever_probe(monkeypatch, captured)

    pipeline.get_retriever(
        vectorstore,
        retrieve_k=8,
        rerank_model="reranker-model",
        rerank_k=3,
        db_directory="./persist",
    )

    assert captured["compressor"] == {
        "model": "encoder:reranker-model",
        "top_n": 3,  # the cross-encoder keeps rerank_k, not the whole budget
    }
    assert captured["compression"]["base_compressor"] == "compressor"


def test_get_retriever_skips_bm25_when_alpha_is_one(monkeypatch):
    captured = {}
    vectorstore = _retriever_probe(monkeypatch, captured)

    pipeline.get_retriever(
        vectorstore,
        retrieve_k=6,
        rerank_model="reranker-model",
        rerank_k=3,
        db_directory="./persist",
        hybrid_alpha=1.0,
    )

    assert captured["dense"]["search_kwargs"] == {"k": 6}
    assert "bm25" not in captured
    assert "ensemble" not in captured
    assert captured["compression"]["base_retriever"] == "dense-retriever"


def test_get_retriever_skips_vector_store_when_alpha_is_zero(monkeypatch):
    captured = {}
    vectorstore = _retriever_probe(monkeypatch, captured)

    pipeline.get_retriever(
        vectorstore,
        retrieve_k=6,
        rerank_model="reranker-model",
        rerank_k=3,
        db_directory="./persist",
        hybrid_alpha=0.0,
    )

    assert captured["bm25"] == {"db_directory": "./persist", "k": 6}
    assert "dense" not in captured
    assert "ensemble" not in captured
    assert captured["compression"]["base_retriever"] == "bm25-retriever"


def test_get_retriever_falls_back_to_dense_when_bm25_index_missing(monkeypatch):
    captured = {}
    vectorstore = _retriever_probe(monkeypatch, captured, bm25="missing")

    result = pipeline.get_retriever(
        vectorstore,
        retrieve_k=6,
        rerank_model="reranker-model",
        rerank_k=3,
        db_directory="./persist",
        hybrid_alpha=0.5,
    )

    assert result == "compressed-retriever"
    # The whole budget falls back to the dense arm, not just its former share.
    assert captured["dense"]["search_kwargs"] == {"k": 6}
    assert "ensemble" not in captured
    assert captured["compression"]["base_retriever"] == "dense-retriever"


def test_get_retriever_narrows_both_arms_to_the_selected_files(monkeypatch):
    """`include_sources` has to reach the dense filter *and* the lexical pass.

    The two arms express a scope differently — Chroma takes a metadata filter,
    BM25 can only be told what it is allowed to return — so a scope that reached
    only one of them would leave the other free to cite a file the user took out
    of context.
    """
    captured = {}
    vectorstore = _retriever_probe(monkeypatch, captured)
    alpha = r"D:\docs\alpha.pdf"

    pipeline.get_retriever(
        vectorstore,
        retrieve_k=10,
        rerank_model="reranker-model",
        rerank_k=3,
        db_directory="./persist",
        hybrid_alpha=0.5,
        include_sources=[alpha],
        all_sources=[alpha],
    )

    assert captured["dense"]["search_kwargs"]["filter"] == {"source": {"$in": [alpha]}}
    assert captured["bm25_sources"] == frozenset({os.path.normcase(alpha)})


def test_get_retriever_excludes_files_from_both_arms(monkeypatch):
    captured = {}
    vectorstore = _retriever_probe(monkeypatch, captured)
    alpha = r"D:\docs\alpha.pdf"
    beta = r"D:\docs\beta.pdf"

    pipeline.get_retriever(
        vectorstore,
        retrieve_k=10,
        rerank_model="reranker-model",
        rerank_k=3,
        db_directory="./persist",
        hybrid_alpha=0.5,
        exclude_sources=[alpha],
        all_sources=[alpha, beta],
    )

    # The filter keeps the corpus's own spelling: the store compares it against
    # the exact string the index recorded, and a folded path would match nothing.
    assert captured["dense"]["search_kwargs"]["filter"] == {"source": {"$nin": [alpha]}}
    assert captured["bm25_sources"] == frozenset({os.path.normcase(beta)})


def test_get_retriever_searches_everything_by_default(monkeypatch):
    captured = {}
    vectorstore = _retriever_probe(monkeypatch, captured)

    pipeline.get_retriever(
        vectorstore,
        retrieve_k=10,
        rerank_model="reranker-model",
        rerank_k=3,
        db_directory="./persist",
        hybrid_alpha=0.5,
    )

    assert "filter" not in captured["dense"]["search_kwargs"]
    assert captured["bm25_sources"] is None


@pytest.mark.parametrize(
    ("include", "exclude", "all_sources", "expected_where", "expected_allowed"),
    [
        # Nothing selected: no filter and no allow-list. An empty `$in` is an
        # error in Chroma, and the caller refuses the question instead of asking
        # it of an empty context. `None` allowed means "unknown, do not filter",
        # which is not the same as "nothing".
        ([], None, None, None, None),
        # A file named in both is excluded: the click that hid it is the more
        # recent instruction.
        ([DOC], [DOC], None, None, frozenset()),
        # An exclusion without the corpus's file list cannot become an exact
        # allow-list, so the lexical arm stays unfiltered rather than guessing.
        (None, [DOC], None, {"source": {"$nin": [DOC]}}, None),
    ],
)
def test_source_filter_edge_cases(
    include, exclude, all_sources, expected_where, expected_allowed
):
    where, allowed = pipeline.source_filter(include, exclude, all_sources)

    assert where == expected_where
    assert allowed == expected_allowed


def test_source_filter_matches_paths_case_insensitively():
    """Path spelling is the browser's, not the index's.

    The allow-list is folded so the two spellings compare equal, and the filter
    is rewritten to the corpus's own spelling — which is the only spelling the
    store will match.
    """
    where, allowed = pipeline.source_filter(
        [r"d:\DOCS\Alpha.PDF"], None, [r"D:\docs\alpha.pdf"]
    )

    assert where == {"source": {"$in": [r"D:\docs\alpha.pdf"]}}
    assert allowed == frozenset({os.path.normcase(r"D:\docs\alpha.pdf")})


def test_source_filter_uses_the_corpus_spelling_in_the_filter():
    """The store matches the filter literally, so the folded path is only for
    comparison — the filter carries the spelling the index recorded."""
    where, _ = pipeline.source_filter(
        None, [r"D:\DOCS\Alpha.PDF"], [r"D:\docs\alpha.pdf"]
    )

    assert where == {"source": {"$nin": [r"D:\docs\alpha.pdf"]}}


def test_format_docs_joins_page_content():
    docs = [SimpleNamespace(page_content="A"), SimpleNamespace(page_content="B")]
    assert pipeline.format_docs(docs) == "A\n\nB"


def test_get_prompt_template_calls_from_messages(monkeypatch):
    captured = {}

    class FakePromptTemplate:
        @staticmethod
        def from_messages(messages):
            captured["messages"] = messages
            return "prompt-template"

    monkeypatch.setattr(pipeline, "ChatPromptTemplate", FakePromptTemplate)

    result = pipeline.get_prompt_template("use context")

    assert result == "prompt-template"
    assert captured["messages"] == [("system", "use context"), ("human", "{question}")]


def test_get_prompt_template_adds_history_placeholder(monkeypatch):
    captured = {}

    class FakePromptTemplate:
        @staticmethod
        def from_messages(messages):
            captured["messages"] = messages
            return "prompt-template"

    monkeypatch.setattr(pipeline, "ChatPromptTemplate", FakePromptTemplate)

    pipeline.get_prompt_template("use context", with_history=True)

    assert captured["messages"][0] == ("system", "use context")
    assert captured["messages"][1].variable_name == "chat_history"
    assert captured["messages"][2] == ("human", "{question}")


def test_condense_question_passthrough_without_history():
    assert pipeline.condense_question([], "plain question", "llm-not-called") == (
        "plain question"
    )


def test_condense_question_rewrites_with_history(monkeypatch):
    captured = {}

    def fake_invoke(messages):
        captured["messages"] = messages
        return SimpleNamespace(content="standalone question")

    class FakeParser:
        def invoke(self, message):
            return message.content

    monkeypatch.setattr(pipeline, "StrOutputParser", lambda: FakeParser())

    history = [("human", "What is the pricing?"), ("ai", "It starts at $10.")]
    fake_llm = SimpleNamespace(invoke=fake_invoke)
    result = pipeline.condense_question(
        history, "what about the annual plan?", fake_llm
    )

    assert result == "standalone question"
    rendered = captured["messages"][1].content
    assert "User: What is the pricing?" in rendered
    assert "Assistant: It starts at $10." in rendered


def test_build_rag_chain_threads_history(monkeypatch):
    captured = {}
    fake_llm = FakeLLM(captured)
    monkeypatch.setattr(pipeline, "get_llm", lambda *a, **k: fake_llm)
    monkeypatch.setattr(
        pipeline,
        "get_retriever",
        lambda *a, **k: SimpleNamespace(
            invoke=lambda q: [SimpleNamespace(page_content="doc")]
        ),
    )

    def fake_condense(history, question, llm):
        captured["history"] = history
        captured["question"] = question
        return "standalone"

    monkeypatch.setattr(pipeline, "condense_question", fake_condense)

    history = [("human", "prev question"), ("ai", "prev answer")]
    chain, _ = pipeline.build_rag_chain(
        vectorstore=object(),
        llm_model="m",
        llm_provider="ollama",
        system_prompt="sys",
        retrieve_k=5,
        llm_temperature=0.0,
        rerank_model="test-reranker",
        rerank_k=3,
        rerank_threshold=0.0,
        chat_history=history,
    )

    out = chain.invoke({"question": "follow-up", "chat_history": history})

    assert out == "final answer"
    assert captured["history"] == history
    assert captured["question"] == "follow-up"
    assert _message_contents(captured["prompt_messages"]) == [
        "sys",
        "prev question",
        "prev answer",
        "follow-up",
    ]


def test_build_rag_chain_without_history_skips_condense(monkeypatch):
    captured = {}
    fake_llm = FakeLLM(captured)
    monkeypatch.setattr(pipeline, "get_llm", lambda *a, **k: fake_llm)
    monkeypatch.setattr(
        pipeline,
        "get_retriever",
        lambda *a, **k: SimpleNamespace(
            invoke=lambda q: [SimpleNamespace(page_content="doc")]
        ),
    )

    def fake_condense(*args):
        captured["condense_called"] = True
        return "condensed"

    monkeypatch.setattr(pipeline, "condense_question", fake_condense)

    chain, _ = pipeline.build_rag_chain(
        vectorstore=object(),
        llm_model="m",
        llm_provider="ollama",
        system_prompt="sys",
        retrieve_k=5,
        llm_temperature=0.0,
        rerank_model="test-reranker",
        rerank_k=3,
        rerank_threshold=0.0,
    )

    out = chain.invoke({"question": "plain", "chat_history": []})

    assert out == "final answer"
    assert "condense_called" not in captured
    assert _message_contents(captured["prompt_messages"]) == ["sys", "plain"]


def test_build_rag_chain_applies_score_threshold(monkeypatch):
    captured = {"threshold": None}
    fake_llm = FakeLLM({})
    monkeypatch.setattr(pipeline, "get_llm", lambda *a, **k: fake_llm)
    monkeypatch.setattr(
        pipeline,
        "get_retriever",
        lambda *a, **k: SimpleNamespace(
            invoke=lambda q: [SimpleNamespace(page_content="doc")]
        ),
    )

    def fake_threshold(docs, threshold):
        captured["threshold"] = threshold
        return docs

    monkeypatch.setattr(pipeline, "filter_by_score_threshold", fake_threshold)

    chain, _ = pipeline.build_rag_chain(
        vectorstore=object(),
        llm_model="m",
        llm_provider="ollama",
        system_prompt="sys",
        retrieve_k=5,
        llm_temperature=0.0,
        rerank_model="test-reranker",
        rerank_k=3,
        rerank_threshold=0.3,
    )

    chain.invoke({"question": "plain", "chat_history": []})

    assert captured["threshold"] == 0.3


def test_reranker_annotates_scores_and_keeps_top_n():
    model = FakeModel([0.1, 0.9, 0.5])
    compressor = pipeline.ScoreAnnotatingReranker(model=model, top_n=2)

    kept = compressor.compress_documents(_docs("A", "B", "C"), "q")

    assert [d.page_content for d in kept] == ["B", "C"]
    assert kept[0].metadata[SCORE_KEY] == 0.9
    assert kept[1].metadata[SCORE_KEY] == 0.5


def test_reranker_sorting_is_descending_by_score():
    model = FakeModel([0.2, 0.8, 0.6])
    compressor = pipeline.ScoreAnnotatingReranker(model=model, top_n=3)

    kept = compressor.compress_documents(_docs("A", "B", "C"), "q")

    assert [d.page_content for d in kept] == ["B", "C", "A"]


def test_threshold_drops_low_scoring_keep_order():
    docs = _docs("A", "B", "C")
    docs[0].metadata[SCORE_KEY] = 0.8
    docs[1].metadata[SCORE_KEY] = 0.2
    docs[2].metadata[SCORE_KEY] = 0.5

    kept = pipeline.filter_by_score_threshold(docs, 0.3)

    assert [d.page_content for d in kept] == ["A", "C"]


def test_threshold_disabled_when_zero_or_none():
    docs = _docs("A", "B")
    docs[0].metadata[SCORE_KEY] = 0.1

    assert pipeline.filter_by_score_threshold(docs, 0.0) == docs
    assert pipeline.filter_by_score_threshold(docs, None) == docs


def test_threshold_fail_open_for_unscored_docs():
    docs = _docs("A", "B")
    docs[0].metadata[SCORE_KEY] = 0.8
    # B has no score -> kept

    kept = pipeline.filter_by_score_threshold(docs, 0.5)

    assert [d.page_content for d in kept] == ["A", "B"]


def test_threshold_keeps_the_best_chunks_rather_than_nothing():
    """An absolute cutoff must never empty the context.

    The cross-encoder's scale is not comparable across queries: corpus- and
    document-level questions score their relevant chunks near zero while content
    questions score near one. Discarding everything there guarantees a "cannot
    answer" reply and hides the retrieved sources.
    """
    docs = _docs("A", "B", "C", "D", "E")
    for doc, score in zip(docs, (0.0077, 0.0006, 0.0002, 0.0001, 0.0000)):
        doc.metadata[SCORE_KEY] = score

    kept = pipeline.filter_by_score_threshold(docs, 0.3)

    # Nothing cleared the bar, so the best few are passed through, in rank order.
    assert [d.page_content for d in kept] == ["A", "B", "C"]
    assert len(kept) <= pipeline._FALLBACK_CONTEXT_CHUNKS


def test_threshold_fallback_is_capped_and_ordered_by_score():
    docs = _docs("low", "high", "mid")
    for doc, score in zip(docs, (0.01, 0.03, 0.02)):
        doc.metadata[SCORE_KEY] = score

    kept = pipeline.filter_by_score_threshold(docs, 0.9)

    # Rank order (as the retriever returned them), not score order.
    assert [d.page_content for d in kept] == ["low", "high", "mid"]


def test_threshold_fallback_does_not_fire_when_something_clears_the_bar():
    docs = _docs("A", "B", "C")
    docs[0].metadata[SCORE_KEY] = 0.9
    docs[1].metadata[SCORE_KEY] = 0.2
    docs[2].metadata[SCORE_KEY] = 0.1

    kept = pipeline.filter_by_score_threshold(docs, 0.3)

    # The threshold still does its job when it has something to keep.
    assert [d.page_content for d in kept] == ["A"]


def test_threshold_fallback_is_empty_for_an_empty_retrieval():
    assert pipeline.filter_by_score_threshold([], 0.5) == []
