from flask import Flask, request, jsonify
import os
import requests
import google.generativeai as genai
import PIL.Image
from PIL import Image
from openai import OpenAI, OpenAIError
from io import BytesIO
import base64
import openai
from dotenv import load_dotenv

load_dotenv(override=True)

app = Flask(__name__)


# Configure API keys
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route('/generate_cartoon_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    file = request.files['image']
    img = Image.open(file.stream)
    
    # Generate a description using Google Generative AI
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(["Please describe the image comprehensively, including every detail necessary for an image generation model to create it accurately.", img])
    description = response.text
    
    client = OpenAI()
    response = client.images.generate(
        model="dall-e-3",
        prompt=f"{description}. Create a 3D cartoon-style image that looks realistic",
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    
    # Download the image to get base64 encoding
    image_content = requests.get(image_url).content
    image_base64 = base64.b64encode(image_content).decode('utf-8')
    return jsonify({
        "image_url": image_url,
        "image_base64": image_base64
    })



if __name__ == '__main__':
    app.run(debug=True)
