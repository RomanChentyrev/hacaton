from typing import Optional

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Reranker:
    def __init__(
        self,
        encoder_model: SentenceTransformer,
        history_df: pl.DataFrame,
        facts_df: pl.DataFrame,
    ):
        self.encoder = encoder_model
        self._history = history_df
        self._facts_db = facts_df

    def rerank(
        self,
        query: str,
        user_id: str,
        topk: Optional[int] = None,
    ) -> pl.DataFrame:
        query_emb = self.encoder.encode([query], prompt_name="search_query")

        fact_ids = (
            pl.Series(
                self._history
                .filter(pl.col("user_id") == user_id)
                .select("fact_id")
            )
            .to_list()
        )

        user_facts = (
            self._facts_db
            .filter(pl.col("id").is_in(fact_ids))
        )

        if not topk:
            topk = len(user_facts)

        emb_matrix = np.vstack(user_facts['embedding'].to_list())

        scores = cosine_similarity(
            query_emb.reshape(1, -1),
            emb_matrix
        ).ravel()
        
        ranked = (
            user_facts
            .with_columns(pl.Series('score', scores))
            .sort('score', descending=True)
            .head(topk)
        )

        return ranked
