import os
import cloudinary
import cloudinary.api
from airflow.models import Variable
from datetime import datetime
import logging
from dateutil import parser

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
    
    # Fetch list of resources from Cloudinary's 'users/' folder
    response = cloudinary.api.resources(type="upload", prefix="users/", max_results=500)
    resources = response['resources']
    
    # Filter only images uploaded after last retrain
    new_images = [
        img for img in resources 
        if parser.isoparse(img['created_at']) > last_retrain_time
    ]
    
    logger = logging.getLogger(__name__)
    logger.info(f"Found {len(new_images)} new images in 'users/' after {last_retrain_time}.")
    # print(f"Found {len(new_images)} new images in 'users/' after {last_retrain_time}.")

    if len(new_images) >= 10:
        return 's3_to_csv'
    else:
        return 'stop_no_data'