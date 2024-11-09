import os
import requests
from PIL import Image
from io import BytesIO
import time

# Define your Hugging Face API token and model name
API_TOKEN = "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"
MODEL_NAME = "stabilityai/stable-diffusion-2-1"  # Can change to another model if desired

# Headers for the API request
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
}

# List of prompts to generate images for
prompts = [
    "A student filling out a scholarship form",
    "Financial aid for university students",
    "Young person looking at financial documents",
    # Add more prompts as needed
]

# Directory to save generated images
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# Function to generate and save images from prompts
def generate_images_from_prompts(prompts):
    for idx, prompt in enumerate(prompts):
        print(f"Generating image for prompt: '{prompt}'")
        
        # Payload for the API
        payload = {
            "inputs": prompt,
            "options": {
                "wait_for_model": True
            }
        }

        try:
            # Send request to Hugging Face API
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
                headers=headers,
                json=payload
            )
            response.raise_for_status()  # Raise error if request failed

            # Process and save the image
            image = Image.open(BytesIO(response.content))
            image_path = os.path.join(output_dir, f"image_{idx+1}.png")
            image.save(image_path)
            print(f"Image saved to {image_path}")
            
            # To avoid hitting rate limits, add a short delay
            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            print(f"Error generating image for prompt '{prompt}': {e}")

# Run the function
generate_images_from_prompts(prompts)
