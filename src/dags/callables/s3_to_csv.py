import cloudinary
import cloudinary.api
import csv
import os

# --- Setup your Cloudinary credentials ---
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)

# --- Parameters ---
FOLDERS = ['Decks/', 'Walls/', 'Pavements/']
OUTPUT_CSV = 'cloudinary_dataset_all.csv'
INCLUDE_VERSION = True

# --- Helper to build Cloudinary image URL ---
def build_cloudinary_url(resource, include_version=True):
    base = f"https://res.cloudinary.com/{cloudinary.config().cloud_name}/image/upload"
    path = resource['public_id'] + '.' + resource['format']
    if include_version:
        return f"{base}/v{resource['version']}/{path}"
    else:
        return f"{base}/{path}"

# --- Main function ---
def generate_dataset_csv():
    print(f"📡 Starting scan for folders: {FOLDERS}")
    all_rows = []

    for folder in FOLDERS:
        print(f"🔍 Scanning folder: {folder}")
        next_cursor = None
        total_in_folder = 0

        while True:
            response = cloudinary.api.resources(
                type="upload",
                prefix=folder,
                max_results=500,
                next_cursor=next_cursor
            )
            resources = response.get('resources', [])
            total_in_folder += len(resources)

            for i, res in enumerate(resources):
                url = build_cloudinary_url(res, INCLUDE_VERSION)
                folder_parts = res['public_id'].split('/')
                if len(folder_parts) >= 2:
                    place = folder_parts[0]
                    label = folder_parts[1]
                else:
                    place = folder
                    label = 'unknown'
                all_rows.append([url, place, label])

                if i % 50 == 0:
                    print(f"  → Processed {i+1}/{total_in_folder} images in current batch...")

            next_cursor = response.get('next_cursor')
            if not next_cursor:
                break

        print(f"✅ {total_in_folder} images found in '{folder}'")

    # Write CSV
    print(f"📝 Writing CSV: {OUTPUT_CSV}")
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'place', 'label'])
        for row in all_rows:
            writer.writerow(row)

    print(f"🎉 Done! Dataset CSV saved to: {OUTPUT_CSV}")

# --- Run ---
if __name__ == "__main__":
    generate_dataset_csv()
