import json
import os


def load_entity_definitions(language="en"):
    """Load entity definitions from the JSON file."""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(
        parent_dir, f"entity_definitions/entity_definitions_{language}.json"
    )
    print(f"input_file: {input_file}")

    try:
        with open(input_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # If file doesn't exist, raise an error
        raise FileNotFoundError(
            f"Entity definitions file not found for language: {language}"
        )
