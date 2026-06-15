import os
import re

from transformers import Pipeline, pipeline

class TopicClassifier:

    def __init__(self, model_name: str = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0") -> None:
        self.pipeline: Pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            token=os.getenv("HF_TOKEN")
        )
    
        self.topic_anchors: dict[str, str] = {
                "Agriculture & Rural Development": "Agriculture & Rural Development (farming, livestock, crops, irrigation, forestry, fishing)",
                "Aid Effectiveness": "Aid Effectiveness (development assistance, grants, donor coordination, official development assistance)",
                "Climate Change": "Climate Change (greenhouse gas emissions, carbon, global warming, climate mitigation, adaptation)",
                "Economy & Growth": "Economy & Growth (GDP, gross domestic product, consumption, capital formation, national accounts, investment, economic growth)",
                "Education": "Education (school, literacy, teachers, primary education, secondary education, training, tertiary, mathematics, reading)",
                "Energy & Mining": "Energy & Mining (electricity, fossil fuels, mining, renewable energy, oil, gas, coal, power, renewable)",
                "Environment": "Environment (biodiversity, natural resources, pollution, conservation, ecosystems, water management)",
                "External Debt": "External Debt (debt service, foreign debt, debt relief, arrears, public debt)",
                "Financial Sector": "Financial Sector (banking, insurance, stock markets, financial access, interest rates, monetary policy)",
                "Gender": "Gender (female, male, women's rights, gender equality, empowerment, maternal health)",
                "Health": "Health (nutrition, food, fruit, sugar, honey, fat, oil, medicine, diseases, mortality, healthcare, hospitals, meat, bread, cereal, vegetables, milk, eggs, cheese, alcohol, tobacco)",
                "Infrastructure": "Infrastructure (transport, telecommunications, machinery, equipment, construction, roads, railways, ports, internet)",
                "Millenium development goals": "Millenium development goals (MDG, global targets, basic human needs, development indicators)",
                "Poverty": "Poverty (poverty headcount, inequality, welfare, social safety nets, low income, vulnerable groups)",
                "Private Sector": "Private Sector (business, entrepreneurship, small and medium enterprises, investment climate, privatization)",
                "Public Sector": "Public Sector (government, public finance, governance, tax, civil service, law and justice)",
                "Science & Technology": "Science & Technology (research and development, R&D, innovation, patents, scientific high-tech)",
                "Social Development": "Social Development (recreation, culture, community development, social inclusion, human rights, culture, conflict prevention)",
                "Social Protection & Labor": "Social Protection & Labor (employment, unemployment, labor market, labor force participation, pensions, social security, child labor, youth idle rate)",
                "Trade": "Trade (export, import, tariff, balance of trade, clothing, footwear, commodities, exporter)",
                "Urban Development": "Urban Development (cities, urbanization, municipal finance, urban housing, slum improvement)"
            }

    def find_primary_topic(
        self,
        name: str,
        topic_list: list[str],
        unique_topics: list[str],
        batch_size: int = 1,
    ) -> str:
        lower_case_name = name.lower()
        clean_name = re.sub(r'^\d+:', '', lower_case_name)
        clean_name = re.sub(r'\s*\([^)]*\)', '', clean_name)

        topic_candidates = topic_list if topic_list else unique_topics

        # if no topic listed, determine from from all topics
        if not topic_candidates:
            return "Uncategorized"
        if len(topic_candidates) == 1:
            return str(topic_candidates[0])

        rich_topic_candidates = [self.topic_anchors.get(t.strip(), t.strip()) for t in topic_candidates]
        
        # Manual Override
        if "gross domestic product" in clean_name or "gdp" in clean_name:
            return "Economy & Growth"
        if "oils" in clean_name or "fats" in clean_name:
            return "Health"
        
        result = self.pipeline(
            clean_name, 
            candidate_labels=rich_topic_candidates,
            hypothesis_template="This dataset measures statistics related to {}.",
            batch_size=batch_size,
        )

        best_rich_label = str(result["labels"][0])

        for original, rich in self.topic_anchors.items():
            if rich.strip().lower() == best_rich_label.strip().lower():
                return original
                
        return best_rich_label

    def find_primary_topics(
        self,
        names: list[str],
        topic_list: list[str],
        unique_topics: list[str],
        batch_size: int,
    ) -> list[str]:
        topic_candidates = topic_list if topic_list else unique_topics

        if not topic_candidates:
            return ["Uncategorized" for _ in names]
        if len(topic_candidates) == 1:
            return [str(topic_candidates[0]) for _ in names]

        rich_topic_candidates = [self.topic_anchors.get(t.strip(), t.strip()) for t in topic_candidates]
        primary_topics: list[str | None] = [None for _ in names]
        model_input_names: list[str] = []
        model_input_indexes: list[int] = []

        for index, name in enumerate(names):
            lower_case_name = name.lower()
            clean_name = re.sub(r'^\d+:', '', lower_case_name)
            clean_name = re.sub(r'\s*\([^)]*\)', '', clean_name)

            # Manual Override
            if "gross domestic product" in clean_name or "gdp" in clean_name:
                primary_topics[index] = "Economy & Growth"
                continue
            if "oils" in clean_name or "fats" in clean_name:
                primary_topics[index] = "Health"
                continue

            model_input_names.append(clean_name)
            model_input_indexes.append(index)

        if model_input_names:
            results = self.pipeline(
                model_input_names,
                candidate_labels=rich_topic_candidates,
                hypothesis_template="This dataset measures statistics related to {}.",
                batch_size=batch_size,
            )

            for index, result in zip(model_input_indexes, results):
                best_rich_label = str(result["labels"][0])
                primary_topics[index] = self._get_original_topic(best_rich_label)

        return [topic if topic is not None else "Uncategorized" for topic in primary_topics]

    def _get_original_topic(self, topic_label: str) -> str:
        for original, rich in self.topic_anchors.items():
            if rich.strip().lower() == topic_label.strip().lower():
                return original

        return topic_label
