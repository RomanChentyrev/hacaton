import asyncio
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
import polars as pl
from gigachat import GigaChat
from sentence_transformers import SentenceTransformer

from submission.solution import Solution

load_dotenv()


async def main():
    giga = GigaChat(
        verify_ssl_certs=False,
        profanity_check=False,
        timeout=6000,
        credentials=os.environ.get("GIGACHAT_TOKEN"),
        scope=os.environ.get("GIGACHAT_SCOPE"),
    )

    model = Solution(
        giga=giga,
        encoder=SentenceTransformer("ai-forever/FRIDA"),
        history_df=pl.read_parquet("../../data/user_history.parquet"),  # SET YOUR PATH HERE
        facts_db=pl.read_parquet("../../data/facts_db.parquet"),  # SET YOUR PATH HERE
    )
    
    user_id = 1
    query = "Какие приложения ты используешь для здоровья?"
    t1_start = perf_counter()

    res = await model.answer(user_id, query)
    
    t1_stop = perf_counter()
    print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)
    print(res)


if __name__ == "__main__":
    asyncio.run(main())
