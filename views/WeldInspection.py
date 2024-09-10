import PIL
import streamlit as st
from ultralytics import YOLO
import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
from io import BytesIO
import time


st.title("Weld Defect Dectector")  # Adding header to sidebar

mode = st.radio(
    "Select source type:",
    ('Video','Image')
)

if mode == "Video":
    
    # Title of the app
    st.title("YOLOv8 Object Detection")

    # Initialize session state for the uploaded file and stop button
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'stop' not in st.session_state:
        st.session_state.stop = False

    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

    # Save the uploaded file to session state
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    # Check if a file is uploaded
    if st.session_state.uploaded_file is not None:
        # Save the uploaded file to a temporary location
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(st.session_state.uploaded_file.read())

        # Load the YOLOv8 model
        try:
            model = YOLO("yolov8n.pt")  # Ensure you have the YOLOv8 weights file
        except Exception as e:
            st.error(f"Error loading YOLO model: {e}")
            st.stop()

        # Open the video file
        cap = cv2.VideoCapture(tfile.name)
       

        # Create a placeholder for the video
        video_placeholder = st.empty()

        # Add a button to start object detection
        if st.button("Detect Objects"):
            st.session_state.stop = False

            # Add a button to stop the video
            if st.button("Stop Video"):
                st.session_state.stop = True

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or st.session_state.stop:
                    break

                # Perform object detection
                results = model(frame)

                # Draw bounding boxes and labels on the frame
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Ensure only one element tensors are converted to scalars
                        class_id = int(box.cls)  # Convert tensor to int
                        confidence = float(box.conf)  # Convert tensor to float
                        label_text = f"Class {class_id} ({confidence:.2f})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Convert frame to bytes
                is_success, buffer = cv2.imencode(".jpg", frame)
                io_buf = BytesIO(buffer)

                # Display the frame with bounding boxes and labels
                video_placeholder.image(io_buf, channels="BGR")

                # Control the frame rate
                time.sleep(0.03)

            cap.release()
    
    
    





    
    
    
    



    
   

else:
        
   
     
     if 'uploaded_image' not in st.session_state:
         st.session_state.uploaded_image = None
     if 'detection_results' not in st.session_state:
         st.session_state.detection_results = None

     model_path = YOLO("yolov8n.pt")

     # File uploader
     source_img = st.file_uploader("Upload an image or take a picture", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

     # Save the uploaded image to session state
     if source_img is not None:
         st.session_state.uploaded_image = PIL.Image.open(source_img)

     confidence = 0.5

     # Creating two columns on the main page
     col1, col2 = st.columns(2)

     # Adding image to the first column if image is uploaded
     with col1:
         if st.session_state.uploaded_image:
             # Adding the uploaded image to the page with a caption
             st.image(st.session_state.uploaded_image,
                      caption="Uploaded Image",
                      use_column_width=True
                      )

     try:
         model = model_path
     except Exception as ex:
         st.error(f"Unable to load model. Check the specified path: {model_path}")
         st.error(ex)

     # Display the Detect Objects button only if an image is uploaded
     if st.session_state.uploaded_image is not None:
         if st.button('Detect Objects'):
             with st.spinner('Detecting objects...'):
                 res = model.predict(st.session_state.uploaded_image, conf=confidence)
                 st.session_state.detection_results = res[0]
                 boxes = st.session_state.detection_results.boxes
                 res_plotted = st.session_state.detection_results.plot()[:, :, ::-1]
                 
                 with col2:
                     st.image(res_plotted, caption='Detected Image', use_column_width=True)
                 
                 # Centralizing and displaying the message based on object detection
                 if len(boxes) == 0:
                     st.markdown(
                         "<div style='text-align: center; background-color: #262730; color: #ecf0f1; padding: 10px; border-radius: 5px;'>"
                         "<strong></strong> No objects detected"
                         "</div>",
                         unsafe_allow_html=True
                     )
                 else:
                     # Extract object names from boxes
                     class_names = [model.names[int(box.cls)] for box in boxes]
                     unique_classes = set(class_names)  # Remove duplicates
                     object_names = ", ".join(unique_classes)
                     st.markdown(
                         f"<div style='text-align: center; background-color: #262730; color: #ecf0f1; padding: 10px; border-radius: 5px;'>"
                         f"<strong>Object Detected:</strong> {object_names}"
                         "</div>",
                         unsafe_allow_html=True
                     )

         


    

            

    
    
    
    
    

            

    
    



    
    
    
    




       
