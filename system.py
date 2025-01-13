import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from IPython.display import display, clear_output
import time

def capture_and_predict_with_input_display(model_path, classes, img_height=30, img_width=30, capture_interval=2):
    """
    Captures images from the webcam, preprocesses them, and predicts their class using a pre-trained model.

    Args:
        model_path (str): Path to the pre-trained model file.
        classes (dict): Dictionary mapping class indices to class names.
        img_height (int): Height to which the image should be resized for prediction.
        img_width (int): Width to which the image should be resized for prediction.
        capture_interval (int): Time interval (in seconds) between capturing frames.
    """
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        while True:
            start_time = time.time()

            # Capture frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break

            # Convert the frame to RGB and display it
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            # Preprocess the frame for prediction
            img = cv2.resize(frame, (img_width, img_height))
            normalized_img = img / 255.0
            input_img = np.expand_dims(normalized_img, axis=0)

            # Make a prediction
            predictions = model.predict(input_img)
            class_idx = np.argmax(predictions)
            class_name = classes.get(class_idx, "Unknown")

            # Display captured image and input for prediction
            clear_output(wait=True)
            print("Captured Image:")
            display(pil_img)

            input_img_display = (normalized_img * 255).astype(np.uint8)
            input_pil_img = Image.fromarray(cv2.cvtColor(input_img_display, cv2.COLOR_BGR2RGB))
            print("Input Image for Prediction (Resized and Preprocessed):")
            display(input_pil_img)

            print(f"Prediction: {class_name}")

            # Wait for the next capture interval
            elapsed_time = time.time() - start_time
            time_to_wait = capture_interval - elapsed_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)

    except KeyboardInterrupt:
        print("Exit")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    classes = { 
        0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 
        2:'Speed limit (50km/h)', 3:'Speed limit (60km/h)', 
        4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
        6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 
        8:'Speed limit (120km/h)', 9:'No passing', 
        10:'No passing veh over 3.5 tons', 
        11:'Right-of-way at intersection', 12:'Priority road', 
        13:'Yield', 14:'Stop', 15:'No vehicles', 
        16:'Veh > 3.5 tons prohibited', 17:'No entry', 
        18:'General caution', 19:'Dangerous curve left', 
        20:'Dangerous curve right', 21:'Double curve',  
        22:'Bumpy road', 23:'Slippery road', 
        24:'Road narrows on the right', 25:'Road work', 
        26:'Traffic signals', 27:'Pedestrians', 
        28:'Children crossing', 29:'Bicycles crossing', 
        30:'Beware of ice/snow', 31:'Wild animals crossing', 
        32:'End speed + passing limits', 33:'Turn right ahead', 
        34:'Turn left ahead', 35:'Ahead only', 
        36:'Go straight or right', 37:'Go straight or left', 
        38:'Keep right', 39:'Keep left', 
        40:'Roundabout mandatory', 41:'End of no passing', 
        42:'End no passing veh > 3.5 tons' 
    }
    capture_and_predict_with_input_display("traffic_sign_classifier.h5", classes)
