from pathlib import Path
import sys

if __package__ in (None, ""):
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from world_bank_topic_classifier.app import run_classification
else:
    from .app import run_classification

def main() -> None:
    run_classification(input_path="indicators.csv", output_path="indicator_topic_mapping.csv")
    

if __name__ == "__main__":
    main()
