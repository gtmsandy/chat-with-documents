import unittest
from types import SimpleNamespace
from unittest.mock import patch, sentinel

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

import rag_utils
from rag_utils import (
    _get_context_retriever_chain,
    create_chat_model,
    get_source_citations,
    to_langchain_messages,
)


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
