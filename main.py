from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the model

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    device_map={"": "cuda"},  # Use "cuda" for NVIDIA GPUs
)

# Load your image

image = Image.open("images/test.png")

# 3. Object Detection

print("Detecting objects:")
objects = model.detect(image, "face")["objects"]
print(objects)
print(f"Found {len(objects)} face(s)")

# 4. Visualize the detected faces
def plot_detections(image, objects, save_path="detected_faces.png"):
    """
    Plot detected objects on the image with bounding boxes
    """
    # Create a copy of the image for drawing
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Get image dimensions
    img_width, img_height = image.size
    
    # Colors for bounding boxes (red, green, blue, etc.)
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
    
    for i, obj in enumerate(objects):
        # Convert normalized coordinates to pixel coordinates
        x_min = int(obj['x_min'] * img_width)
        y_min = int(obj['y_min'] * img_height)
        x_max = int(obj['x_max'] * img_width)
        y_max = int(obj['y_max'] * img_height)
        
        # Choose color
        color = colors[i % len(colors)]
        
        # Draw bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
        
        # Add label
        label = f"Face {i+1}"
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
        
        # Draw label background
        text_bbox = draw.textbbox((x_min, y_min - 20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        
        # Draw label text
        draw.text((x_min, y_min - 20), label, fill='white', font=font)
    
    # Save the image
    img_with_boxes.save(save_path)
    print(f"Image with detections saved as: {save_path}")
    
    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(img_with_boxes)
    plt.axis('off')
    plt.title(f'Detected {len(objects)} face(s)')
    plt.tight_layout()
    plt.show()

# Plot the detections
plot_detections(image, objects)
