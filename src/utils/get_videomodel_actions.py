import os
import json
import urllib.request


def clean_json_keys(data: dict):
    """
    Удаляет лишние кавычки внутри ключей вроде "\"eating ice cream\"".
    Возвращает новый словарь.
    """
    cleaned = {}
    for k, v in data.items():
        if isinstance(k, str) and k.startswith('"') and k.endswith('"'):
            k = k.strip('"')
        cleaned[k] = v
    return cleaned


def get_slowfast_classes():
    """
    Download Kinetics-400 class names if not exists and return set of lowercase names.
    Handles both formats: {id: class_name} or {class_name: id}.
    Also cleans invalid quoted keys if found.
    """
    base_dir = os.path.dirname(__file__)
    kinetics_path = os.path.abspath(os.path.join(base_dir, "../../data/external/kinetics_classnames.json"))

    os.makedirs(os.path.dirname(kinetics_path), exist_ok=True)

    # --- download if not exists ---
    if not os.path.exists(kinetics_path):
        print(f"Downloading Kinetics-400 class names to {kinetics_path}...")
        url = "https://raw.githubusercontent.com/facebookresearch/pytorchvideo/main/pytorchvideo/data/kinetics/kinetics_classnames.json"
        urllib.request.urlretrieve(url, kinetics_path)
        print("Download complete.")

    # --- load and clean ---
    with open(kinetics_path, "r", encoding="utf-8") as f:
        kinetics_classes = json.load(f)

    kinetics_classes = clean_json_keys(kinetics_classes)

    # --- detect format ---
    sample_key, sample_value = list(kinetics_classes.items())[0]
    if isinstance(sample_value, str):
        # {"0": "abseiling"}
        class_names = [v.lower() for v in kinetics_classes.values() if isinstance(v, str)]
    else:
        # {"abseiling": 0}
        class_names = [k.lower() for k in kinetics_classes.keys() if isinstance(k, str)]

    # --- save cleaned version nicely ---
    clean_path = kinetics_path.replace(".json", "_clean.json")
    with open(clean_path, "w", encoding="utf-8") as f:
        json.dump(kinetics_classes, f, indent=4, ensure_ascii=False)

    class_names = sorted(set(class_names))
    print(f"Loaded {len(class_names)} classes from Kinetics-400.")
    print("Example classes:", class_names[:10])
    print(f"Cleaned file saved to: {clean_path}")
    return class_names


if __name__ == "__main__":
    get_slowfast_classes()
