import os
from pathlib import Path
import pandas as pd
from dataclasses import asdict

from .data_processing import extract_lists, get_unique_topics
from .model import CategorizedIndicator
from .ml import TopicClassifier
from .observations import split_observations_by_topic
from .utils import check_env_boolean_required

def run_classification(
    input_path: str = "indicators.csv",
    output_path: str = "indicator_topic_mapping.csv",
) -> list[CategorizedIndicator]:
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
    no_topic_batch_size = int(os.getenv("TOPIC_CLASSIFIER_BATCH_SIZE", "50"))

    single_topic_indicators = [indicator for indicator in indicators if len(indicator.topics) == 1]
    no_topic_indicators = [indicator for indicator in indicators if len(indicator.topics) == 0]
    multiple_topic_indicators = [indicator for indicator in indicators if len(indicator.topics) > 1]

    for indicator in single_topic_indicators:
        print("categorizing for indicator: ", indicator.name)
        primary_topic = indicator.topics[0]
        categorized_indicator = CategorizedIndicator(indicator.id,
                                indicator.name,
                                indicator.source_id,
                                indicator.source,
                                indicator.source_organization,
                                primary_topic)
        categorized_indicators.append(categorized_indicator)

    for indicator in multiple_topic_indicators:
        print("categorizing for indicator: ", indicator.name)
        primary_topic = topic_classifier.find_primary_topic(
            indicator.name,
            indicator.topics,
            indicator.topics,
            batch_size=1,
        )
        categorized_indicator = CategorizedIndicator(indicator.id,
                                indicator.name,
                                indicator.source_id,
                                indicator.source,
                                indicator.source_organization,
                                primary_topic)
        categorized_indicators.append(categorized_indicator)
    
    no_topic_primary_topics = topic_classifier.find_primary_topics(
        [indicator.name for indicator in no_topic_indicators],
        [],
        unique_topic_list,
        batch_size=no_topic_batch_size,
    )
    for indicator, primary_topic in zip(no_topic_indicators, no_topic_primary_topics):
        print("categorizing for indicator: ", indicator.name)
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

    return categorized_indicators

def run_etl(input_path: str, output_path: str) -> None:
    reclassify_flag = check_env_boolean_required("RECLASSIFY_FLAG")

    if reclassify_flag:
        categorized_indicators = run_classification(input_path=input_path, output_path=output_path)
    
    if not Path(output_path).is_file():
            raise RuntimeError(f"ERROR: reclassify records or provide: {output_path}")
 
