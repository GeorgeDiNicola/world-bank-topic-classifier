import pandas as pd
from dataclasses import asdict

from .data_processing import extract_lists, get_unique_topics
from .model import CategorizedIndicator
from .ml import TopicClassifier

def run_classification(input_path: str, output_path: str) -> None:
    try:
        indicators_df = pd.read_csv(input_path)
    except FileNotFoundError as e:
        print("the file of indicators does not exist")
        raise RuntimeError(f"Required input file missing: {input_path}") from e
    except pd.errors.EmptyDataError as e:
        print(f"ERROR: The file '{input_path}' was found, but it is empty.")
        raise RuntimeError(f"Required input file is empty: {input_path}") from e
    
    indicators = extract_lists(indicators_df)
    unique_topic_list = get_unique_topics(indicators)
    unique_topic_list.remove("Millenium development goals")

    categorized_indicators = []
    topic_classifier = TopicClassifier()

    for indicator in indicators:
        print("categorizing for indicator: ", indicator.name)
        primary_topic = topic_classifier.find_primary_topic(indicator.name, indicator.topics, unique_topic_list)
        categorized_indicator = CategorizedIndicator(indicator.id,
                                  indicator.name,
                                  indicator.source_id,
                                  indicator.source,
                                  indicator.source_organization,
                                  primary_topic)
        categorized_indicators.append(categorized_indicator)


    records = [asdict(item) for item in categorized_indicators]
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False, encoding='utf-8')