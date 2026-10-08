import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch, sentinel
import uuid

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

import rag_utils
from rag_utils import (
    _get_context_retriever_chain,
    clear_chat_state,
    clear_document_state,
    create_chat_model,
    get_session_chroma_path,
    get_session_source_dir,
    get_source_citations,
    to_langchain_messages,
)


class DocumentLifecycleTests(unittest.TestCase):
    def test_session_paths_are_scoped_and_invalid_session_is_rejected(self):
        session_id = str(uuid.uuid4())
        with TemporaryDirectory() as runtime_dir:
            runtime_path = Path(runtime_dir).resolve()
            self.assertEqual(
                get_session_chroma_path(session_id, runtime_path),
                runtime_path / f"chroma_db_{session_id}",
            )
            self.assertEqual(
                get_session_source_dir(session_id, runtime_path),
                runtime_path / "source_files" / session_id,
            )
            with self.assertRaises(ValueError):
                get_session_chroma_path("../unrelated", runtime_path)

    def test_document_reset_is_session_scoped_and_allows_reupload(self):
        session_id = str(uuid.uuid4())
        other_session_id = str(uuid.uuid4())
        with TemporaryDirectory() as runtime_dir:
            chroma_path = get_session_chroma_path(session_id, runtime_dir)
            source_path = get_session_source_dir(session_id, runtime_dir)
            other_path = get_session_chroma_path(other_session_id, runtime_dir)
            for path in (chroma_path, source_path, other_path):
                path.mkdir(parents=True)
                (path / "marker.txt").write_text("test", encoding="utf-8")

            state = {
                "session_id": session_id,
                "vector_db": object(),
                "rag_sources": ["example.pdf"],
                "use_rag": True,
                "rag_url": "https://example.com",
                "rag_docs_key": 2,
                "messages": [{
                    "role": "assistant",
                    "content": "Grounded answer",
                    "sources": [{"source": "example.pdf", "page": "1"}],
                    "evidence": [{"page_content": "evidence", "metadata": {}}],
                }],
            }

            self.assertEqual(clear_document_state(state, runtime_dir), [])

            self.assertNotIn("vector_db", state)
            self.assertEqual(state["rag_sources"], [])
            self.assertFalse(state["use_rag"])
            self.assertEqual(state["rag_url"], "")
            self.assertEqual(state["rag_docs_key"], 3)
            self.assertNotIn("sources", state["messages"][0])
            self.assertNotIn("evidence", state["messages"][0])
            self.assertFalse(chroma_path.exists())
            self.assertFalse(source_path.exists())
            self.assertTrue(other_path.exists())
            self.assertNotIn("example.pdf", state["rag_sources"])

    def test_invalid_session_refuses_cleanup_without_resetting_state(self):
        state = {
            "session_id": "../unrelated",
            "vector_db": sentinel.vector_db,
            "rag_sources": ["example.pdf"],
        }

        with TemporaryDirectory() as runtime_dir:
            unrelated_path = Path(runtime_dir) / "unrelated"
            unrelated_path.mkdir()
            with self.assertRaises(ValueError):
                clear_document_state(state, runtime_dir)

            self.assertTrue(unrelated_path.exists())
            self.assertIs(state["vector_db"], sentinel.vector_db)
            self.assertEqual(state["rag_sources"], ["example.pdf"])

    def test_locked_runtime_directory_is_reported_after_state_reset(self):
        session_id = str(uuid.uuid4())
        state = {
            "session_id": session_id,
            "rag_sources": ["example.pdf"],
            "use_rag": True,
            "messages": [],
        }

        with TemporaryDirectory() as runtime_dir:
            chroma_path = get_session_chroma_path(session_id, runtime_dir)
            chroma_path.mkdir()
            with patch("rag_utils.shutil.rmtree", side_effect=PermissionError):
                failed_paths = clear_document_state(state, runtime_dir)

            self.assertEqual(failed_paths, [chroma_path])
            self.assertEqual(state["rag_sources"], [])
            self.assertFalse(state["use_rag"])

    def test_clear_chat_does_not_change_document_state(self):
        state = {
            "messages": [{"role": "user", "content": "Hello"}],
            "vector_db": sentinel.vector_db,
            "rag_sources": ["example.pdf"],
            "use_rag": True,
        }

        clear_chat_state(state)

        self.assertEqual(state["messages"], [])
        self.assertIs(state["vector_db"], sentinel.vector_db)
        self.assertEqual(state["rag_sources"], ["example.pdf"])
        self.assertTrue(state["use_rag"])


class ProviderRoutingTests(unittest.TestCase):
    def test_groq_namespaced_model_uses_groq_client(self):
        with (
            patch("rag_utils.ChatGroq") as groq_client,
            patch("rag_utils.ChatOpenAI") as openai_client,
            patch("rag_utils.ChatAnthropic") as anthropic_client,
        ):
            model = create_chat_model(
                "Groq",
                "openai/gpt-oss-20b",
                "gsk_test_key",
            )

        self.assertIs(model, groq_client.return_value)
        groq_client.assert_called_once_with(
            api_key="gsk_test_key",
            model="openai/gpt-oss-20b",
            temperature=0,
            streaming=True,
        )
        self.assertNotIn("base_url", groq_client.call_args.kwargs)
        openai_client.assert_not_called()
        anthropic_client.assert_not_called()

    def test_rag_stages_reuse_the_same_provider_model(self):
        session_state = SimpleNamespace(vector_db=sentinel.vector_db)

        with (
            patch.object(rag_utils, "st", SimpleNamespace(session_state=session_state)),
            patch("rag_utils._get_context_retriever_chain") as retriever_chain,
            patch("rag_utils.create_stuff_documents_chain") as answer_chain,
            patch("rag_utils.create_retrieval_chain") as rag_chain,
        ):
            result = rag_utils.get_conversational_rag_chain(sentinel.provider_model)

        retriever_chain.assert_called_once_with(
            sentinel.vector_db,
            sentinel.provider_model,
        )
        self.assertIs(answer_chain.call_args.args[0], sentinel.provider_model)
        rag_chain.assert_called_once_with(
            retriever_chain.return_value,
            answer_chain.return_value,
        )
        self.assertIs(result, rag_chain.return_value)


class SourceCitationTests(unittest.TestCase):
    def test_citations_use_filename_human_page_numbers_and_deduplication(self):
        documents = [
            Document(page_content="first", metadata={"source": r"source_files\guide.pdf", "page": 11}),
            Document(page_content="duplicate", metadata={"source": r"source_files\guide.pdf", "page": 11}),
            Document(page_content="second", metadata={"source": "notes.md"}),
        ]

        self.assertEqual(
            get_source_citations(documents),
            [
                {"source": "guide.pdf", "page": "12"},
                {"source": "notes.md", "page": None},
            ],
        )


class ChatHistoryTests(unittest.TestCase):
    def test_rich_stored_messages_become_text_only_history(self):
        stored_messages = [
            {"role": "user", "content": "Which topic contains the example?"},
            {
                "role": "assistant",
                "content": "The example is in the allocation topic.",
                "sources": [{"source": "guide.pdf", "page": "12"}],
                "evidence": [{"page_content": "private evidence", "metadata": {}}],
            },
        ]

        history = to_langchain_messages(stored_messages)

        self.assertIsInstance(history[0], HumanMessage)
        self.assertIsInstance(history[1], AIMessage)
        self.assertEqual(
            [message.content for message in history],
            [
                "Which topic contains the example?",
                "The example is in the allocation topic.",
            ],
        )
        self.assertNotIn("private evidence", history[1].content)
        self.assertNotIn("guide.pdf", history[1].content)

    def test_history_aware_retriever_uses_contextualized_query(self):
        prompts = []
        retrieved_queries = []

        class FakeVectorDB:
            def as_retriever(self):
                return RunnableLambda(
                    lambda query: retrieved_queries.append(query) or []
                )

        def contextualize(prompt):
            prompts.append(prompt.to_messages())
            return AIMessage(content="other examples in the allocation topic")

        retriever = _get_context_retriever_chain(
            FakeVectorDB(),
            RunnableLambda(contextualize),
        )
        history = [
            HumanMessage(content="Which topic contains the example?"),
            AIMessage(content="The example is in the allocation topic."),
        ]

        retriever.invoke({
            "input": "What are the other examples in that same topic?",
            "chat_history": history,
        })

        self.assertEqual(retrieved_queries, ["other examples in the allocation topic"])
        self.assertEqual(
            [message.content for message in prompts[0]][-3:],
            [
                "Which topic contains the example?",
                "The example is in the allocation topic.",
                "What are the other examples in that same topic?",
            ],
        )


if __name__ == "__main__":
    unittest.main()
