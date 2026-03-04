import os
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import numpy as np

'''
# ----- Configuration -----
# Path to your input image (adjust as needed)
input_image_path = "2022-10-12_solalinden_waldwege__0095_1665579514737884132_windshield_vis.png"

# Directory to save generated images
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# The prompt is used to guide the generation process.
# You can modify this prompt to better describe the type of image you want.
prompt = "A similar image"

# Strength controls how much the model deviates from the input image.
# Lower values preserve more of the original image.
strength = 0.5

# Guidance scale controls how closely the generated image follows the prompt.
guidance_scale = 7.5

# Number of images to generate
num_images = 10

# ----- Load the Input Image -----
# Open the image and ensure it is in RGB mode.
init_image = Image.open(input_image_path).convert("RGB")
# Resize the image to 512x512 (the typical resolution for many diffusion models).
init_image = init_image.resize((512, 512))

init_image_np = np.array(init_image)

init_image_list = [init_image]

# ----- Load the Diffusion Model -----
# Here we use the Stable Diffusion model from runwayml.
# The model is loaded with half precision (fp16) for performance if supported.
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token="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"
)

# Use GPU if available, else CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# ----- Generate Similar Images -----
print("Starting generation...")
for i in range(num_images):
    # Generate a new image from the initial image.
    result = pipe(prompt=prompt, init_image=init_image_np, strength=strength, guidance_scale=guidance_scale)
    generated_image = result.images[0]

    # Save the generated image.
    output_path = os.path.join(output_dir, f"generated_{i + 1}.png")
    generated_image.save(output_path)
    print(f"Saved: {output_path}")

print("Generation complete.")

'''

import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "2022-10-12_solalinden_waldwege__0095_1665579514737884132_windshield_vis.png"
init_image = load_image(url)

prompt = "photorealistic, forest road with intersection, create completely new image with new directions "

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, guidance_scale=15, strength=0.2).images[0]

image.save("TEST.png")