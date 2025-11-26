import csv
import time
import os  # Necessary to handle paths universally
from huggingface_hub import HfApi

# 1. Initialize API
api = HfApi()

# 2. List of features (keywords) to search for
features = ["agriculture", "crop", "biomass", "yield", "nitrogen", "phosphorus", "soil", "plant", "drone", "remote sensing", "time series", "field", "tomato", "heatmap", "organism", "vegetable"]

# 3. Retrieve all unique IDs and track the found feature
model_features_map = {}  # model_id -> list of found features

print(f"Starting search for features: {features}")

for feature in features:
    print(f"🔍 Searching for: {feature}...")
    try:
        # Search by name and by tag
        ids_by_name = {m.modelId for m in api.list_models(search=feature, limit=None)}
        ids_by_tag  = {m.modelId for m in api.list_models(filter={"tags": [feature]}, limit=None)}
        matching_ids = ids_by_name | ids_by_tag  # Union of sets

        for model_id in matching_ids:
            if model_id not in model_features_map:
                model_features_map[model_id] = []
            model_features_map[model_id].append(feature)
            
    except Exception as e:
        print(f"⚠️ Error searching for '{feature}': {e}")

print(f"Found {len(model_features_map)} unique models.")

# 4. Prepare CSV with universal path
# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Create the full path for the CSV file (works on Win, Mac, Linux)
csv_file = os.path.join(current_dir, "hf_filtered_models.csv")

fields = [
    "id", "downloads", "likes", "tags",
    "pipeline_tag", "library_name",
    "license", "model_type", "last_modified",
    "feature_found"
]

print(f"Writing data to: {csv_file}")

with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    # Counter for visual feedback
    count = 0
    total = len(model_features_map)

    for model_id, features_found in model_features_map.items():
        count += 1
        # Print progress every 50 models
        if count % 50 == 0:
            print(f"Processing {count}/{total}...")

        try:
            # Retrieve basic information
            info = api.model_info(model_id)
            downloads = getattr(info, "downloads", 0) # Default to 0 if missing
            likes = getattr(info, "likes", 0)
            tags = info.tags or []
            pipeline = info.pipeline_tag or ""
            library = info.library_name or ""
            last_mod = info.lastModified or ""

            # Metadata from cardData
            card = info.cardData or {}
            license_ = card.get("license", "")
            model_type = card.get("model_type", "")

            # Write to CSV
            writer.writerow({
                "id": model_id,
                "downloads": downloads,
                "likes": likes,
                "tags": ", ".join(tags),
                "pipeline_tag": pipeline,
                "library_name": library,
                "license": license_,
                "model_type": model_type,
                "last_modified": last_mod,
                "feature_found": ", ".join(features_found)
            })

            # Sleep to avoid 429 errors (Rate Limit)
            time.sleep(0.5)

        except Exception as e:
            print(f"⚠️ Error retrieving info for {model_id}: {e}")
            continue

print(f"✅ Done! CSV saved at: {csv_file}")