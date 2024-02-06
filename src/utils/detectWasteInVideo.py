import cv2
import numpy as np
from ultralytics import YOLO  # Replace with the actual YOLO library import if different

def generate_colors(num_classes):
    """
    Generate distinct colors for each class.
    This uses a seeded RNG for reproducibility.
    """
    np.random.seed(42)  # Seed for reproducibility
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_classes)]
    return colors

def detect_waste_in_video(video_path):
    # Load the model
    model = YOLO("/Users/pramudithakarunarathna/Documents/IIT Final Year/FYP/Implementations/wastePileDetection/models/best.pt")  # Update the path to your model

    # Assuming model has an attribute 'names' that contains class labels
    num_classes = len(model.names)  # Adjust according to your model
    # Generate unique colors for each class
    colors = generate_colors(num_classes)

    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Get frame rate and set up output video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection on the frame
        results = model.predict(frame)
        result = results[0]

        # Draw bounding boxes with unique color per class
        for box in result.boxes:
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            class_id = box.cls[0].item()  # Ensure this is correctly fetching the class ID
            prob = round(box.conf[0].item(), 2)
            color = colors[int(class_id)]  # Use class ID as integer to select color

            # Draw rectangle and text with class-specific color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{result.names[int(class_id)]}: {prob}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Write frame to output video and display
        out.write(frame)
        cv2.imshow('Waste Detection', frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Call the function with your video path
detect_waste_in_video("/Users/pramudithakarunarathna/Documents/IIT Final Year/FYP/Implementations/wastePileDetection/Test/videos/IMG_4799.MOV")
