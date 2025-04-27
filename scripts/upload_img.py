import os
import time
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables``
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)

root_folder = 'Dataset'
MAX_WORKERS = 3  # <= Important! Fewer workers = less risk
DELAY_BETWEEN_UPLOADS = 0.4  # <= 0.4s between uploads = 2.5 uploads/sec

def upload_image(file_path, relative_path, public_id, idx, total_files):
    print(f"[{idx}/{total_files}] Uploading {file_path} as {relative_path}/{public_id}...")
    try:
        result = cloudinary.uploader.upload(
            file_path,
            folder=relative_path,
            public_id=public_id,
            unique_filename=False,  # Keep filename
            overwrite=False,        # Skip if already exists
            resource_type="image"
        )
        time.sleep(DELAY_BETWEEN_UPLOADS)  # 🛡️ Safe throttle
        return result
    except Exception as e:
        print(f"❗ Error uploading {file_path}: {e}")
        return None

for subdir, _, files in os.walk(root_folder):
    relative_path = os.path.relpath(subdir, root_folder).replace("\\", "/")
    parts = relative_path.split('/')

    if len(parts) == 2:  # Only Decks/Cracked etc.
        valid_files = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])

        if not valid_files:
            continue

        total_files = len(valid_files)
        print(f"\n🛫 Found {total_files} images in {relative_path}...\n")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for idx, file in enumerate(valid_files, start=1):
                file_path = os.path.join(subdir, file)
                filename_without_ext = os.path.splitext(file)[0]

                futures.append(executor.submit(
                    upload_image,
                    file_path,
                    relative_path,
                    filename_without_ext,
                    idx,
                    total_files
                ))

            for future in as_completed(futures):
                _ = future.result()

print("\n🚀 Upload complete.")
