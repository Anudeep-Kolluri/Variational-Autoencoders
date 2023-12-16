import tkinter as tk
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

from vae_code import get_model

vae, decoder = get_model()


def cvt_range(val):
    val /= 10
    mid = val / 2
    return round(mid - 7, 2)


def on_hover(event):
    x, y = cvt_range(event.x), cvt_range(event.y)
    coordinates_var.set(f"X: {x}, Y: {y}")

    # Fetch the MNIST image based on the coordinates
    mnist_image = decoder.predict([[x, y]])[0][:, :, 0] * 255

    # Create a larger version of the image
    larger_image = Image.fromarray(mnist_image).resize((200, 200))

    # Convert the PIL Image to Tkinter PhotoImage
    mnist_image_tk = ImageTk.PhotoImage(larger_image)

    # Update the image on the main window
    mnist_label.configure(image=mnist_image_tk)
    mnist_label.image = mnist_image_tk

    # Update coordinates in the second window
    update_second_window(x, y)


def update_second_window(x, y):
    # Fetch the MNIST image based on the coordinates
    decoded_image = decoder.predict([[x, y]])[0]

    # Scale the values to the range [0, 255] and convert to uint8
    decoded_image = (decoded_image * 255).astype(np.uint8)

    # Create a larger version of the image
    larger_image = Image.fromarray(decoded_image)

    # Resize the image to (200, 200)
    larger_image = larger_image.resize((200, 200))

    # Convert the PIL Image to Tkinter PhotoImage
    image_tk = ImageTk.PhotoImage(larger_image)

    # Update the image on the second window
    mnist_label.configure(image=image_tk)
    mnist_label.image = image_tk

    # Update coordinates in the second window
    coordinates_var.set(f"X: {x}, Y: {y}")


# Create the main window
window = tk.Tk()
window.title("CelebA Variational Autoencoder")

# Create a frame to hold the canvas and labels
frame = tk.Frame(window)
frame.pack(padx=10, pady=10)

# Create a canvas with a 2D grid background
canvas = tk.Canvas(frame, width=300, height=300, bg="white")
canvas.grid(row=0, column=0, padx=10, pady=10)

# Create a label to display the mouse coordinates
coordinates_var = tk.StringVar()
coordinates_label = tk.Label(frame, textvariable=coordinates_var, font=("Helvetica", 12))
coordinates_label.grid(row=1, column=0, pady=5)

# Draw the 2D grid on the canvas
for i in range(0, 300, 30):
    canvas.create_line(i, 0, i, 300, fill="lightgray")
    canvas.create_line(0, i, 300, i, fill="lightgray")

# Create a label to display the MNIST image
mnist_label = tk.Label(frame)
mnist_label.grid(row=0, column=1, padx=10, pady=10)

# Bind the hover event to the canvas
canvas.bind("<Motion>", on_hover)

# Run the Tkinter event loop
window.mainloop()
