import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import time

# Define the custom metric function
def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)

# Register the custom metric function
custom_objects = {"dice_coef": dice_coef}

# Global variables
model = None
model_path = ""

# Create the GUI window
window = tk.Tk()
window.title("Polyp Segmentation Using IRv2-Net Model")
window.geometry("1000x500")

# Create a label for displaying frame rate
frame_rate_label = tk.Label(window, text="Frame Rate: N/A", font=("Arial", 15))
frame_rate_label.pack()

# Function to import the model H5 file
def import_model():
    global model, model_path

    # Open file dialog to select a model H5 file
    model_path = filedialog.askopenfilename(filetypes=[("Model files", "*.h5")])

    if model_path:
        # Load the selected model H5 file
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path)
        print("Model imported successfully!")

# Function to process the image and generate the mask
def process_image():
    if not model:
        print("Please import a model first!")
        return

    start_time = time.time()

    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        # Load the selected image
        image = Image.open(file_path)

        # Resize the image to 256x256
        image = image.resize((256, 256))

        # Convert image to numpy array
        image_array = np.array(image)

        # Normalize the image array
        image_array = image_array / 255.0

        # Add an extra dimension to match the model's input shape
        image_array = np.expand_dims(image_array, axis=0)

        # Generate the mask using the loaded model
        mask = model.predict(image_array)

        # Rest of your image processing code

        end_time = time.time()
        processing_time = end_time - start_time

        if processing_time > 0:
            frame_rate = 1.0 / processing_time
            frame_rate_label.config(text=f"Frame Rate: {frame_rate:.2f} FPS")
        else:
            frame_rate_label.config(text="Frame Rate: Infinity FPS")

        # Display the original image, ground truth mask, and predicted mask in the GUI
        # ...

# Create a button to import the model
import_model_button = tk.Button(window, text="Import Model", command=import_model, font=("Arial", 15))
import_model_button.pack()

# Create a button to process the image
process_button = tk.Button(window, text="Process Image", command=process_image, font=("Arial", 15))
process_button.pack()

# Create labels to display the original image, ground truth mask, and predicted mask
image_label = tk.Label(window)
image_label.pack(side=tk.LEFT)

ground_truth_label = tk.Label(window)
ground_truth_label.pack(side=tk.LEFT)

predicted_mask_label = tk.Label(window)
predicted_mask_label.pack(side=tk.LEFT)

# Start the GUI event loop
window.mainloop()
