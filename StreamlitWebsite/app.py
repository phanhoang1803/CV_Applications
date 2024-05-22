import streamlit as st
from tasks.dog_classification.DogClassification import DogClassification
from tasks.car_detection.CarDetection import CarDetection
from tasks.edge_detection.EdgeDetection import EdgeDetection
from tasks.face_recog.face_recog import FaceRecognition
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import os

pages = {
    'Home': 'home',
    'Dog/Cat Classification': 'dog_cat_classification_task',
    'Car Detection': 'car_detection_task',
    'Edge Detection': 'edge_detection_task',
    'Local Features Detection': 'local_features_detection_task',
    'Face Recognition': 'face_recognition_task',
}

# Initialize models
@st.cache_resource
def get_dog_classifier():
    return DogClassification()

# Cache the CarDetection model
@st.cache_resource
def get_car_detector():
    return CarDetection()

database_file = "face_database.pkl"
@st.cache_resource
def get_face_recognizer():
    return FaceRecognition(database_file=database_file)

st.sidebar.title('Navigation')
selected_page = st.sidebar.radio('Pages', options=list(pages.keys()))

def home():
    st.title('AI Applications Website')
    st.text('This website is developed by Hoang Phan.')

def dog_cat_classification_task():
    st.header('Dog or Cat Classification')
    
    uploaded_image = st.file_uploader('Please upload a dog or cat image to classify.', type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        with st.spinner('Processing...'):
            # Preprocess the image
            img = image.load_img(uploaded_image, target_size=(224, 224))
            img = tf.expand_dims(img, axis=0)
            
            # Get the cached DogClassification model
            dog_classifier = get_dog_classifier()
            
            # Make prediction
            result = dog_classifier.predict(img)
            
            # Display the uploaded image
            st.image(uploaded_image, caption=f'{result}', use_column_width=True)
        
def car_detection_task():
    st.header('Car Detection')
    
    uploaded_image = st.file_uploader('Please upload a car image to detect.', type=["jpg", "jpeg", "png"])        
    if uploaded_image is not None:
        with st.spinner('Processing...'):
            # Preprocess the image
            with open("temp_uploaded_image.jpg", "wb") as f:
                f.write(uploaded_image.getbuffer())
            
            # Get the cached CarDetection model
            car_detector = get_car_detector()
            
            # Make prediction
            scores, boxes, classes, output_image_path = car_detector.predict("temp_uploaded_image.jpg")
            
            # Display the uploaded image
            st.image(output_image_path, caption='Detected cars', use_column_width=True)
        
def edge_detection_task():
    st.header('Edge Detection')
    
    selected_method = st.selectbox("Select edge detection method", ["Sobel", "Prewitt", "Laplacian", "Canny"])
    
    uploaded_image = st.file_uploader('Please upload a car image to detect.', type=["jpg", "jpeg", "png"])      
    if uploaded_image is not None:
        with st.spinner('Processing...'):
            img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
            # img = uploaded_image.frombuffer()
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            gray_img = cv2.GaussianBlur(gray_img,(3,3),0)
            
            if selected_method == "Sobel":
                res = EdgeDetection.detectBySobel(gray_img)
            elif selected_method == "Prewitt":
                res = EdgeDetection.detectByPrewitt(gray_img)
            elif selected_method == "Laplacian":
                res = EdgeDetection.detectByLaplacian(gray_img)
            elif selected_method == "Canny":
                res = EdgeDetection.detectByCanny(gray_img)
            
            # Display original image and detected edges side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption='Original Image', use_column_width=True)
            with col2:
                st.image(res, caption='Detected Edges', use_column_width=True)
            
def local_features_detection_task():
    st.header('Local Features Detection')
    
    # ['Harris', 'NLoG', 'DoG', 'SIFT']
    selected_method = st.selectbox('Select local features detection method', ["SIFT", "ORB", "BRISK", "KAZE"])
    
    uploaded_image = st.file_uploader('Please upload an image to detect local features.', type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Initialize feature detector based on selected method
        st.write("Processing...")
        
        if selected_method == "SIFT":
            detector = cv2.SIFT_create()
        elif selected_method == "ORB":
            detector = cv2.ORB_create()
        elif selected_method == "BRISK":
            detector = cv2.BRISK_create()
        elif selected_method == "KAZE":
            detector = cv2.KAZE_create()
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = detector.detectAndCompute(gray_img, None)
        
        # Draw keypoints on the original image
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Remove processing message and display images
        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption='Original Image', use_column_width=True)
        with col2:
            st.image(img_with_keypoints, caption='Image with Detected Features', use_column_width=True)

def face_recognition_task():
    st.header('Face Recognition')
    face_recognizer = get_face_recognizer()

    # Register faces for face recognition
    st.subheader('Register faces for face recognition')
    register_images = st.file_uploader("Please upload face image for face recognition", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    if st.button('Register') and register_images:
        with st.spinner('In progress...'):
            for image in register_images:
                label = os.path.splitext(image.name)[0]
                img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
                success = face_recognizer.register_face(img, label, database_file=database_file)
                if success:
                    st.success("Images added to database successfully!")
                else:
                    st.error("Failed to add images to database!")
    
    # Display all registered faces
    st.subheader('Display all registered faces')
    if st.button('Display'):
        empty = True
        for i, (face_encoding, label) in enumerate(face_recognizer.get_registered_faces(database_file)):
            st.text(f'Face {i+1}: {label}')
            empty = False
        if empty:
            st.text(f'Database is empty. Please register more faces.')
    
    # Recognize faces
    st.subheader('Recognize faces')    
    options = st.selectbox('Choose Input Options:', ["Image Upload", "Real-time Video"])
    if options == "Image Upload":
        # st.subheader('Upload Image to Recognize')
        
        uploaded_image = st.file_uploader("Upload Image for Face Recognition", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_image is not None:
            with st.spinner('In progress...'):
                img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
                
                # Recognize faces
                face_locations, names = face_recognizer.recognize_faces(img)

                # Draw labels on faces
                for (top, right, bottom, left), name in zip(face_locations, names):
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(img, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
                # Display image
                st.image(img, channels='BGR', caption='Recognized Faces')
    elif options == "Real-time Video":
        # st.subheader('Real-time Video Face Recognition')
        
        video_capture = cv2.VideoCapture(0)
        frame_location = st.empty()
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            
            # Recognize faces
            face_locations, names = face_recognizer.recognize_faces(frame)
            
            # Draw labels on faces
            for (top, right, bottom, left), name in zip(face_locations, names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            frame_location.image(frame, channels="BGR", caption="Real-time Face Recognition")
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the video capture
        video_capture.release()
        cv2.destroyAllWindows()
        
    
    # Upload 
    pass
    
if selected_page in pages:
    eval(pages[selected_page])()
