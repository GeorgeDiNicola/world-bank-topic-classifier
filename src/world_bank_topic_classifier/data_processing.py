from typing import Any, cast

import pandas as pd

from .model import Indicator

def extract_lists(df: pd.DataFrame) -> list[Indicator]:
    df['topics'] = (
        df['topics']
        .fillna("")
        .astype(str)
        .str.strip()
        .str.split(r'\s*,\s*')   # Splits by comma, absorbing spaces on either side
    )

    # Handle empty strings
    df['topics'] = df['topics'].apply(lambda x: [] if x == [''] else x)

    raw_records = df.to_dict(orient='records')
    records = cast(list[dict[str, Any]], raw_records)
    return [Indicator(**record) for record in records]

def get_unique_topics(indicators: list[Indicator]) -> list[str]:
    all_topics = {topic for indicator in indicators for topic in indicator.topics}
    return list(set(all_topics))
