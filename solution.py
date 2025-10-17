import polars as pl
from sentence_transformers import SentenceTransformer

from gigachat import GigaChat
from .prompts import RAG_PROMPT
from .reranker import Reranker


class Solution:
    """
    Solution class to answer questions about user.
    """
    def __init__(
        self,
        giga: GigaChat,
        encoder: SentenceTransformer,
        history_df: pl.DataFrame,
        facts_db: pl.DataFrame
    ) -> None:
        """
        Args:
            giga (gigachat.GigaChat): Instance of GigaChat API model.
            encoder (sentence_transformers.SentenceTransformer): Encoder model.
            history_df (pl.DataFrame): Polars dataframe with histories.
            facts_db (pl.DataFrame): Polars dataframe with facts info.
        """
        self._llm = giga
        # encoder = SentenceTransformer('ai-forever/FRIDA')
        self._reranker = Reranker(encoder, history_df, facts_db)
        self._prompt = RAG_PROMPT
        self._topk_facts = 30

    def _get_ranked_facts(self, user_id: str, question: str) -> str:
        reranked_facts = self._reranker.rerank(query=question, user_id=user_id, topk=self._topk_facts)

        relevant_facts_content = reranked_facts["content"].to_list()

        return "\n".join(relevant_facts_content)

    async def answer(self, user_id: str, question: str) -> str:
        """Async method to answer question about user.
        Args:
            user_id (str) User ID
            question (str) Question
        
        Returns:
            str: Answer
        """
        relevant_facts_str = self._get_ranked_facts(user_id, question)
        response = await self._llm.achat(
            self._prompt.format(query=question, facts=relevant_facts_str)
        )

        return response.choices[0].message.content
