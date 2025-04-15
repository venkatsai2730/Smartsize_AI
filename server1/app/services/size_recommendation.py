from loguru import logger

class SizeRecommender:
    def recommend_size(self, measurements: dict) -> str:
        try:
            chest = measurements.get("chest", 85)
            size_chart = {
                "XS": (65, 80),
                "S": (80, 90),
                "M": (90, 100),
                "L": (100, 110),
                "XL": (110, 120),
                "XXL": (120, 135),
            }
            for size, (min_c, max_c) in size_chart.items():
                if min_c <= chest < max_c:
                    return size
            return "XXXL" if chest >= 135 else "XS" if chest < 65 else "Unknown"
        except Exception as e:
            logger.error(f"Size recommendation failed: {e}")
            return "Unknown"

    def validate_size(self, predicted_size: str, ground_truth_size: str) -> dict:
        metrics = {}
        is_correct = predicted_size == ground_truth_size
        metrics["size_match"] = is_correct
        metrics["accuracy"] = 100.0 if is_correct else 0.0
        logger.info(f"Size: Predicted={predicted_size}, Ground Truth={ground_truth_size}, "
                   f"Match={is_correct}, Accuracy={metrics['accuracy']:.2f}%")
        return metrics