import os
import cloudinary
import cloudinary.api
from airflow.models import Variable
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configure your Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)

def check_new_images(**kwargs):
    # Load the last retraining time
    last_retrain_str = Variable.get("last_retrain_time", default_var="2023-01-01T00:00:00")
    last_retrain_time = datetime.fromisoformat(last_retrain_str)
    
    # Fetch list of resources from Cloudinary
    response = cloudinary.api.resources(type="upload", prefix="Decks/", max_results=500)
    resources = response['resources']
    
    # Filter only images uploaded after last retrain
    new_images = [
        img for img in resources 
        if datetime.fromtimestamp(img['created_at']) > last_retrain_time
    ]
    
    print(f"Found {len(new_images)} new images after {last_retrain_time}.")

    if len(new_images) >= 50:
        return 'merge_datasets'  # The next task id if enough data
    else:
        return 'stop_no_data'   # DummyOperator if not enough

