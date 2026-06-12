import pytest

from world_bank_topic_classifier.ml import TopicClassifier


@pytest.fixture(scope="module")
def classifier() -> TopicClassifier:
    return TopicClassifier()


@pytest.mark.parametrize(
    ("name", "topic_list", "unique_topics", "expected"),
    [
        # Test classifier returns uncategorized when no topics available
        ("Population ages 15-64", [], [], "Uncategorized"),
        # Test classifier returns the only available topic
        ("School enrollment, primary", ["Education"], ["Education"], "Education"),
        # Test classifier returns economy label for GDP growth
        (
            "GDP growth (annual %)",
            ["Trade", "Economy & Growth", "Health"],
            ["Trade", "Economy & Growth", "Health"],
            "Economy & Growth",
        ),
        # Test classifier returns Health for oils & fats
        (
            "Food oils and fats exports (% of merchandise exports)",
            ["Trade", "Health"],
            ["Trade", "Health"],
            "Health",
        ),
        # Test the primary topic is found when multiple options in topic list
        (
            "Literacy rate, adult total (% of people ages 15 and above)",
            ["Education", "Health", "Gender"],
            ["Education", "Health", "Gender"],
            "Education",
        ),
        # Test the primary topic is found when there is no topic list
        (
            "Merchandise exports (current US$)",
            [],
            ["Health", "Trade", "Education"],
            "Trade",
        ),
        ("Sample indicator", [], ["Climate Change"], "Climate Change"),
    ],
)
def test_find_primary_topic(
    classifier: TopicClassifier,
    name: str,
    topic_list: list[str],
    unique_topics: list[str],
    expected: str,
) -> None:
    result = classifier.find_primary_topic(
        name=name,
        topic_list=topic_list,
        unique_topics=unique_topics,
    )

    assert result == expected
