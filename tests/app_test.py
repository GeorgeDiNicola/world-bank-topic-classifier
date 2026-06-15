from pathlib import Path

import pandas as pd

from world_bank_topic_classifier import app


class FakeTopicClassifier:

    def __init__(self) -> None:
        self.single_calls: list[dict[str, object]] = []
        self.batch_calls: list[dict[str, object]] = []

    def find_primary_topic(
        self,
        name: str,
        topic_list: list[str],
        unique_topics: list[str],
        batch_size: int = 1,
    ) -> str:
        self.single_calls.append(
            {
                "name": name,
                "topic_list": topic_list,
                "unique_topics": unique_topics,
                "batch_size": batch_size,
            }
        )
        return "Poverty"

    def find_primary_topics(
        self,
        names: list[str],
        topic_list: list[str],
        unique_topics: list[str],
        batch_size: int,
    ) -> list[str]:
        self.batch_calls.append(
            {
                "names": names,
                "topic_list": topic_list,
                "unique_topics": unique_topics,
                "batch_size": batch_size,
            }
        )
        return ["Energy & Mining" for _ in names]


def write_indicator_csv(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "id": "single-topic",
                "name": "Primary completion rate",
                "source_id": "1",
                "source": "World Bank",
                "source_organization": "World Bank",
                "topics": "Education",
            },
            {
                "id": "multiple-topics",
                "name": "Poverty gap at national poverty lines",
                "source_id": "2",
                "source": "World Bank",
                "source_organization": "World Bank",
                "topics": "Poverty, Economy & Growth",
            },
            {
                "id": "no-topic",
                "name": "Access to electricity",
                "source_id": "3",
                "source": "World Bank",
                "source_organization": "World Bank",
                "topics": "",
            },
            {
                "id": "mdg-topic",
                "name": "Millennium development goal sample",
                "source_id": "4",
                "source": "World Bank",
                "source_organization": "World Bank",
                "topics": "Millenium development goals",
            },
        ]
    ).to_csv(path, index=False)


def test_run_classification_batches_no_topic_indicators(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "indicators.csv"
    output_path = tmp_path / "indicator_topic_mapping.csv"
    fake_classifier = FakeTopicClassifier()

    write_indicator_csv(input_path)
    monkeypatch.setenv("TOPIC_CLASSIFIER_BATCH_SIZE", "5")
    monkeypatch.setattr(app, "TopicClassifier", lambda: fake_classifier)

    categorized_indicators = app.run_classification(
        input_path=str(input_path),
        output_path=str(output_path),
    )

    # ensure topics are in the expected order
    assert [indicator.topic for indicator in categorized_indicators] == [
        "Education",
        "Millenium development goals",
        "Poverty",
        "Energy & Mining",
    ]
    assert len(fake_classifier.batch_calls) == 1
    batch_call = fake_classifier.batch_calls[0]
    assert batch_call["names"] == ["Access to electricity"]
    assert batch_call["topic_list"] == []
    assert set(batch_call["unique_topics"]) == {"Education", "Poverty", "Economy & Growth"}
    assert batch_call["batch_size"] == 5


def test_run_classification_uses_batch_size_one_for_multiple_topics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "indicators.csv"
    output_path = tmp_path / "indicator_topic_mapping.csv"
    fake_classifier = FakeTopicClassifier()

    write_indicator_csv(input_path)
    monkeypatch.setattr(app, "TopicClassifier", lambda: fake_classifier)

    app.run_classification(
        input_path=str(input_path),
        output_path=str(output_path),
    )

    assert fake_classifier.single_calls == [
        {
            "name": "Poverty gap at national poverty lines",
            "topic_list": ["Poverty", "Economy & Growth"],
            "unique_topics": ["Poverty", "Economy & Growth"],
            "batch_size": 1,
        }
    ]


def test_run_classification_writes_topic_mapping_csv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "indicators.csv"
    output_path = tmp_path / "indicator_topic_mapping.csv"
    fake_classifier = FakeTopicClassifier()

    write_indicator_csv(input_path)
    monkeypatch.setattr(app, "TopicClassifier", lambda: fake_classifier)

    app.run_classification(
        input_path=str(input_path),
        output_path=str(output_path),
    )

    mapping_df = pd.read_csv(output_path)

    assert list(mapping_df["id"]) == [
        "single-topic",
        "mdg-topic",
        "multiple-topics",
        "no-topic",
    ]
    assert list(mapping_df["topic"]) == [
        "Education",
        "Millenium development goals",
        "Poverty",
        "Energy & Mining",
    ]
