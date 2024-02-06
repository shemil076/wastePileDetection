# # from ultralytics import YOLO

# # def detect_objects_on_image2(image):
# #     model = YOLO("/Users/pramudithakarunarathna/Documents/IIT Final Year/FYP/Implementations/wastePileDetection/models/best.pt")
# #     results = model.predict(image)
# #     result = results[0]
# #     output = []
# #     for box in result.boxes:
# #         x1, y1, x2, y2 = [
# #             round(x) for x in box.xyxy[0].tolist()
# #         ]
# #         class_id = box.cls[0].item()
# #         prob = round(box.conf[0].item(), 2)
# #         output.append([
# #             x1, y1, x2, y2, result.names[class_id], prob
# #         ])
# #     return output

# # detect_objects_on_image2('/Users/pramudithakarunarathna/Documents/IIT Final Year/FYP/Implementations/wastePileDetection/data/test/images/waste_02_jpg.rf.9c8e95e581a540d88341b4efc85f2ee2.jpg')


# --------------------------------------------------

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from collections import deque

# class SmoothedBox:
#     def __init__(self, max_length=10):
#         self.max_length = max_length
#         self.buffer = deque(maxlen=max_length)

#     def update(self, new_box):
#         self.buffer.append(new_box)
#         return self.get_smoothed_box()

#     def get_smoothed_box(self):
#         if not self.buffer:
#             return None
#         return np.mean(self.buffer, axis=0).astype(int)

# def real_time_waste_detection():
#     model = YOLO("/Users/pramudithakarunarathna/Documents/IIT Final Year/FYP/Implementations/wastePileDetection/models/best.pt")
#     cap = cv2.VideoCapture(0)
#     smoothed_boxes_dict = {}

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Perform detection
#         results = model.predict(frame)
#         result = results[0]

#         for box in result.boxes:
#             x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
#             class_id = box.cls[0].item()
#             prob = round(box.conf[0].item(), 2)

#             # Unique identifier for each class of object
#             object_id = (class_id, box.cls[0].item())

#             if object_id not in smoothed_boxes_dict:
#                 smoothed_boxes_dict[object_id] = SmoothedBox()

#             smoothed_box = smoothed_boxes_dict[object_id].update([x1, y1, x2, y2])

#             # Use 'smoothed_box' for drawing
#             if smoothed_box is not None:
#                 x1, y1, x2, y2 = smoothed_box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{result.names[class_id]}: {prob}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

#         cv2.imshow('Real-Time Waste Detection', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# real_time_waste_detection()

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from collections import deque

# class SmoothedBox:
#     def __init__(self, max_length=10):
#         self.max_length = max_length
#         self.buffer = deque(maxlen=max_length)

#     def update(self, new_box):
#         self.buffer.append(new_box)
#         return self.get_smoothed_box()

#     def get_smoothed_box(self):
#         if not self.buffer:
#             return None
#         return np.mean(self.buffer, axis=0).astype(int)

# def real_time_waste_detection():
#     model = YOLO("/Users/pramudithakarunarathna/Documents/IIT Final Year/FYP/Implementations/wastePileDetection/models/best.pt")
#     cap = cv2.VideoCapture(0)
#     smoothed_boxes_dict = {}

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Perform detection
#         results = model.predict(frame)
#         result = results[0]

#         for box in result.boxes:
#             x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
#             class_id = box.cls[0].item()
#             prob = round(box.conf[0].item(), 2)

#             # Unique identifier for each class of object
#             object_id = (class_id, box.cls[0].item())

#             if object_id not in smoothed_boxes_dict:
#                 smoothed_boxes_dict[object_id] = SmoothedBox()

#             smoothed_box = smoothed_boxes_dict[object_id].update([x1, y1, x2, y2])

#             # Use 'smoothed_box' for drawing
#             if smoothed_box is not None:
#                 x1, y1, x2, y2 = smoothed_box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{result.names[class_id]}: {prob}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

#         cv2.imshow('Real-Time Waste Detection', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# real_time_waste_detection()

# --------------------------------------------------
import cv2
from ultralytics import YOLO

# def detect_objects_on_image2(image_path):
#     # Load the model
#     model = YOLO("/Users/pramudithakarunarathna/Documents/IIT Final Year/FYP/Implementations/wastePileDetection/models/best.pt")
    
#     # Load the image
#     img = cv2.imread(image_path)

#     # Perform detection
#     results = model.predict(img)
#     result = results[0]

#     # Draw bounding boxes
#     for box in result.boxes:
#         x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
#         class_id = box.cls[0].item()
#         prob = round(box.conf[0].item(), 2)

#         # Draw rectangle on the image
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(img, f"{result.names[class_id]}: {prob}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

#     # Display the image
#     cv2.imshow("Detected Objects", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from ultralytics import YOLO  # Assuming YOLO is a class from a module you have access to

# def generate_colors(num_classes):
#     """
#     Generate distinct colors for each class.
#     This uses a seeded RNG for reproducibility.
#     """
#     np.random.seed(42)  # Seed for reproducibility
#     colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_classes)]
#     return colors

# def detect_objects_on_image2(image_path):
#     """
#     Detect objects in an image and draw bounding boxes with different colors for each class.
#     """
#     # Initialize the model
#     model = YOLO("/Users/pramudithakarunarathna/Documents/IIT Final Year/FYP/Implementations/wastePileDetection/models/best1.pt")
    
#     # Load the image
#     img = cv2.imread(image_path)

#     # Perform detection
#     results = model.predict(img)
#     result = results[0]

#     # Generate colors for each class
#     colors = generate_colors(19)  # Adjust this if the number of classes changes

#     # Draw bounding boxes
#     for box in result.boxes:
#         x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
#         class_id = int(box.cls[0].item())  # Cast class_id to int to avoid TypeError
#         prob = round(box.conf[0].item(), 2)

#         # Select color based on class ID
#         color = colors[class_id]

#         # Draw rectangle on the image
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(img, f"{result.names[class_id]}: {prob}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     # Display the image
#     cv2.imshow("Detected Objects", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# Note: Ensure the path to your YOLO model is correctly specified.


# Remember to replace "/path/to/your/model.pt" with the actual path to your model.


# Call the function with the image path




# def detect_waste_in_video(video_path):
#     # Load the model
#     model = YOLO("/Users/pramudithakarunarathna/Documents/IIT Final Year/FYP/Implementations/wastePileDetection/models/best.pt")

#     # Open the video
#     cap = cv2.VideoCapture(video_path)

#     # Prepare output video writer (Optional)
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Perform detection on the frame
#         results = model.predict(frame)
#         result = results[0]

#         # Draw bounding boxes
#         for box in result.boxes:
#             x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
#             class_id = box.cls[0].item()
#             prob = round(box.conf[0].item(), 2)

#             # Draw rectangle on the frame
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"{result.names[class_id]}: {prob}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

#         # Display the frame
#         cv2.imshow('Waste Detection', frame)

#         # Write frame to output video (Optional)
#         out.write(frame)

#         # Break loop with 'q' key
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release resources
#     cap.release()
#     out.release()  # If you are writing to output
#     cv2.destroyAllWindows()

# Call the function with the video path
    

# detect_objects_on_image2('/Users/pramudithakarunarathna/Documents/IIT Final Year/FYP/Implementations/wastePileDetection/Test/images/test2.jpeg')

# detect_waste_in_video('/Users/pramudithakarunarathna/Documents/IIT Final Year/FYP/Implementations/wastePileDetection/Test/videos/IMG_4797.MOV')


# import cv2
# from ultralytics import YOLO  # Replace with actual YOLO library import

# import cv2
# # Assuming YOLO and other necessary imports are done correctly

# def detect_waste_in_video(video_path):
#     # Load the model
#     model = YOLO("/Users/pramudithakarunarathna/Documents/IIT Final Year/FYP/Implementations/wastePileDetection/models/best1.pt")  # Update the path to your model

#     # Open the video
#     cap = cv2.VideoCapture(video_path)

#     # Check if video opened successfully
#     if not cap.isOpened():
#         print("Error opening video stream or file")
#         return

#     # Get frame rate of the input video
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # Prepare output video writer
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Perform detection on the frame
#         results = model.predict(frame)
#         result = results[0]

#         # Generate colors for each class
#         colors = generate_colors(19)  # Adjust this if the number of classes changes


#         # Draw bounding boxes
#         for box in result.boxes:
#             x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
#             class_id = box.cls[0].item()
#             prob = round(box.conf[0].item(), 2)

#             # Draw rectangle on the frame
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"{result.names[class_id]}: {prob}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

#         # Write frame to output video
#         out.write(frame)

#         # Display the frame
#         cv2.imshow('Waste Detection', frame)

#         # Break loop with 'q' key
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release resources
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

    
# detect_waste_in_video("/Users/pramudithakarunarathna/Documents/IIT Final Year/FYP/Implementations/wastePileDetection/Test/videos/IMG_4799.MOV")


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
