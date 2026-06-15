# World Bank Topic Classifier

A Python application for classifying World Bank indicators into a single primary topic using zero-shot natural language classification (machine learning).

The World Bank indicator catalog contains many indicators with missing topic metadata, while others are assigned to multiple overlapping topics. This project standardizes those indicators into one primary topic per indicator so they can be used more consistently in downstream data science, analytics, and feature engineering workflows.

## Running the Application
```
make install
make run
```
## Design

### Motivation

The World Bank indicator dataset is not uniformly labeled. Some indicators have no topic assignment, some have exactly one topic, and others are associated with multiple topic categories. Since the goal of this project is to produce a single primary topic for each indicator, the application treats these cases differently rather than applying the same inference path to the entire dataset.

Before designing the classification pipeline, I analyzed the frequency distribution of topic assignments across all 29,512 indicators:

- 23,936 indicators have no topic assigned
- 4,934 indicators have exactly one topic assigned
- 642 indicators have multiple topics assigned

This distribution has an important consequence for the system design: most indicators require model inference, but a meaningful subset can be classified deterministically without invoking the NLP model.

### Indicator Topic Distribution
![Indicator Topic Distribution](<design/Indicator Topic Histogram.png>)

#### Indicator Topic Histogram without Missing Values

![Indicator Topic Distribution](<design/Indicator Topic Histogram Without Missing Values.png>)

## Classification Methodology

The classification pipeline partitions the indicator dataset into three groups:

1. Indicators with no topic
2. Indicators with exactly one topic
3. Indicators with multiple topics

Each group is handled according to the amount of topic metadata available.

### 1. Indicators with No Topic

Indicators with no assigned topic require full model-based classification. These records are passed through the NLP model with the complete list of possible World Bank topics as candidate labels.

Because this group contains the majority of the dataset, it benefits the most from batching. The application uses Hugging Face’s native batching support to reduce inference overhead and improve throughput.

### 2. Indicators with a Single Topic

Indicators with exactly one existing topic do not require model inference. Since the available metadata already maps the indicator to a single category, the pipeline classifies these records deterministically using the existing topic.

This avoids unnecessary model calls, reduces execution time, and prevents the model from reclassifying data that already has an unambiguous topic assignment.

### 3. Indicators with Multiple Topics

Indicators with multiple topics require a more constrained classification step. Instead of choosing from the full set of possible topics, the model selects the most appropriate primary topic from the indicator’s existing topic set.

This approach preserves the signal already present in the World Bank metadata while still reducing each indicator to one primary topic.

## Batching Strategy

The batching strategy is based on the observed frequency distribution of topic assignments.

The indicators with no topic are batched together because they all use the same candidate topic list. This allows the Hugging Face model to process a large number of records with a consistent input structure.

The indicators with multiple topics are handled separately because each record may have a different set of candidate topics. Hugging Face’s native batching is only effective when the batch shares the same candidate label structure. Since the multi-topic records have heterogeneous candidate topic lists, batching them together would add implementation complexity without providing a marginal runtime benefit.

This is especially true because the multi-topic group is relatively small. Topic combinations are granular, and each non-singular topic combination contains at most 36 indicators, with most combinations containing between 1 and 10 indicators. As a result, the complexity of building additional batching logic for these small heterogeneous groups would likely exceed the performance benefit.

## Execution Strategy

The application uses a metadata-aware natural language zero shot classification strategy:

- Automatically classify single-topic indicators using their existing topic
- Batch no-topic indicators using the full candidate topic list
- Individually classify multi-topic indicators using their existing candidate topic set
- Avoid unnecessary model inference when the topic assignment is already deterministic

This design reduces inference cost while preserving classification quality.

## Benchmarking

The project compares execution time between two approaches:

1. Sequential inference
   - Each indicator is processed independently.
   - This is simple, but inefficient for large groups of records with the same candidate labels.
   - Sequential inference would be more acceptable if most indicators could be classified deterministically. In this dataset, however, more than 80% of indicators have no assigned topic, so most records still require model inference.

2. Partitioned batching
   - Indicators are partitioned based on topic metadata.
   - Large homogeneous groups are batched.
   - Heterogeneous multi-topic records are processed with customized candidate labels.
   - Deterministic records are classified without model inference.


### Execution Time Results

| Strategy | Runtime | Notes |
|---|---:|---|
| Sequential inference | 306.92 minutes (~5.12 hours) | Processes each indicator independently, with deterministic classification where possible |
| Partitioned batching, batch size 50 | 118 minutes (~1.97 hours) | ~6 GB RAM peak usage |
| Partitioned batching, batch size 100 | 118 minutes (~1.97 hours) | ~11 GB RAM peak usage |
| Partitioned batching, batch size 500 | 134 minutes (~2.23 hours) | ~34 GB RAM peak usage |

## Hardware

Benchmarks were run locally on:

- MacBook Pro
- 36 GB RAM
- Apple M3 Pro chip

## Model Selection

This project uses MoritzLaurer/deberta-v3-large-zeroshot-v2.0 for zero-shot topic classification.

I chose this model because the World Bank indicator dataset does not provide a complete labeled training set for the target classification task. Many indicators have missing topic metadata, and others have multiple possible topics rather than a single ground-truth label. A zero-shot classifier is a good fit for this problem because it can assign labels from a provided candidate topic list without requiring task-specific fine-tuning.

The model is designed for zero-shot classification through the Hugging Face pipeline and can run on both CPU and GPU hardware. It uses a Natural Language Inference formulation, where each candidate topic is evaluated as a possible label for the input text. This makes it appropriate for a metadata normalization task where the label space is known, but labeled examples are incomplete or inconsistent.

The model also provides a practical tradeoff between classification quality and implementation complexity. Instead of training a custom supervised classifier, this project can use the existing World Bank topic taxonomy directly as candidate labels. That keeps the pipeline simpler, avoids creating a hand-labeled training dataset, and makes the classification process easier to adapt if the set of World Bank topics changes later.

In this project, the model is used in two ways:

* Indicators with no topic are classified against the full World Bank topic list.
* Indicators with multiple topics are classified against their existing topic candidates to select one primary topic.

## Results

Using the frequency distribution of topic assignments to drive the batching strategy reduced unnecessary model inference and improved classification throughput.

Key results:

- Single-topic indicators are classified without model inference
- No-topic indicators are batched efficiently using a shared candidate topic list
- Multi-topic indicators are classified against a smaller, metadata-derived candidate set
- The pipeline is expected to scale with stronger hardware, especially for the batched no-topic inference workload
- Partitioned batching reduced ML classification runtime by up to **61.55%** compared with sequential inference.

## Summary

This project converts inconsistently labeled World Bank indicator metadata into a standardized primary-topic classification. The design combines deterministic rules with NLP-based inference, using the structure of the dataset itself to reduce runtime and avoid unnecessary model calls.

The main engineering decision is to avoid treating all indicators uniformly. Instead, the pipeline uses the available metadata to choose the least expensive classification method that still produces a consistent primary topic.