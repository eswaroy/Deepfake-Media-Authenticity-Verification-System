"""
Deepfake Media Authenticity Verification System
This system detects deepfake videos using deep learning techniques.
Dataset: UADFV (University at Albany DeepFake Video Dataset)
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DeepfakeDetector:
    def __init__(self, base_path, image_size=(224, 224), batch_size=32):
        """
        Initialize the deepfake detector.
        
        Args:
            base_path (str): Path to the UADFV dataset
            image_size (tuple): Input image dimensions (height, width)
            batch_size (int): Batch size for training
        """
        self.base_path = Path(base_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = None
        self.history = None
        
    def load_metadata(self, metadata_file='metadata.csv'):
        """
        Load or create dataset metadata.
        """
        metadata_path = self.base_path / metadata_file
        
        if os.path.exists(metadata_path):
            print(f"Loading existing metadata from {metadata_path}")
            return pd.read_csv(metadata_path)
        
        print("Creating new metadata file...")
        
        # Assuming UADFV structure:
        # - fake/: contains fake videos
        # - real/: contains real videos
        fake_dir = self.base_path / 'fake'
        real_dir = self.base_path / 'real'
        
        data = []
        
        # Process fake videos
        if fake_dir.exists():
            for video_path in fake_dir.glob('*.mp4'):
                data.append({
                    'video_path': str(video_path),
                    'label': 1  # 1 for fake
                })
        
        # Process real videos
        if real_dir.exists():
            for video_path in real_dir.glob('*.mp4'):
                data.append({
                    'video_path': str(video_path),
                    'label': 0  # 0 for real
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(metadata_path, index=False)
        print(f"Metadata saved to {metadata_path}")
        
        return df
    
    def extract_faces_from_video(self, video_path, output_dir, max_frames=20):
        """
        Extract face frames from a video.
        
        Args:
            video_path (str): Path to the video
            output_dir (str): Directory to save extracted faces
            max_frames (int): Maximum number of frames to extract
        
        Returns:
            list: Paths to extracted face images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        video_name = os.path.basename(video_path).split('.')[0]
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine frame sampling rate to get roughly max_frames
        if total_frames <= max_frames:
            frame_indices = range(total_frames)
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        face_image_paths = []
        
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            # Process each face in the frame
            for face_idx, (x, y, w, h) in enumerate(faces):
                # Add some margin around the face
                margin = int(0.2 * w)
                x_start = max(0, x - margin)
                y_start = max(0, y - margin)
                x_end = min(frame.shape[1], x + w + margin)
                y_end = min(frame.shape[0], y + h + margin)
                
                face_image = frame[y_start:y_end, x_start:x_end]
                
                if face_image.size == 0:
                    continue
                
                # Resize the face image
                face_image = cv2.resize(face_image, self.image_size)
                
                # Save the face image
                face_image_path = os.path.join(output_dir, f"{video_name}_frame{i}_face{face_idx}.jpg")
                cv2.imwrite(face_image_path, face_image)
                face_image_paths.append(face_image_path)
                
        cap.release()
        return face_image_paths
    
    def preprocess_dataset(self, metadata_df, faces_dir):
        """
        Preprocess the dataset by extracting faces from videos.
        
        Args:
            metadata_df (DataFrame): Metadata containing video paths and labels
            faces_dir (str): Directory to save extracted faces
        
        Returns:
            DataFrame: Updated metadata with face image paths
        """
        os.makedirs(faces_dir, exist_ok=True)
        
        real_dir = os.path.join(faces_dir, 'real')
        fake_dir = os.path.join(faces_dir, 'fake')
        
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)
        
        face_data = []
        
        for idx, row in metadata_df.iterrows():
            video_path = row['video_path']
            label = row['label']
            
            print(f"Processing {idx+1}/{len(metadata_df)}: {video_path}")
            
            # Determine output directory based on label
            output_dir = fake_dir if label == 1 else real_dir
            
            # Extract faces from video
            face_paths = self.extract_faces_from_video(video_path, output_dir)
            
            # Add each face to the dataset
            for face_path in face_paths:
                face_data.append({
                    'video_path': video_path,
                    'face_path': face_path,
                    'label': label
                })
        
        face_df = pd.DataFrame(face_data)
        
        # Save the face metadata
        face_metadata_path = os.path.join(faces_dir, 'face_metadata.csv')
        face_df.to_csv(face_metadata_path, index=False)
        
        return face_df
    
    def load_face_image(self, face_path):
        """
        Load and preprocess a face image.
        
        Args:
            face_path (str): Path to the face image
            
        Returns:
            ndarray: Preprocessed face image
        """
        img = cv2.imread(face_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, self.image_size)
        img = img / 255.0  # Normalize to [0, 1]
        return img
    
    def create_data_generators(self, face_df):
        """
        Create training and validation data generators.
        
        Args:
            face_df (DataFrame): DataFrame containing face image paths and labels
            
        Returns:
            tuple: (train_generator, val_generator, X_test, y_test)
        """
        # Split the data into training, validation, and test sets
        train_df, temp_df = train_test_split(face_df, test_size=0.3, stratify=face_df['label'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        # Create data generators
        train_generator = self.create_generator(train_df)
        val_generator = self.create_generator(val_df)
        
        # Load test data
        X_test = np.array([self.load_face_image(path) for path in test_df['face_path']])
        y_test = np.array(test_df['label'])
        
        return train_generator, val_generator, X_test, y_test
    
    def create_generator(self, df):
        """
        Create a data generator from a DataFrame.
        
        Args:
            df (DataFrame): DataFrame containing face image paths and labels
            
        Returns:
            Generator: Data generator
        """
        def generator():
            indices = np.arange(len(df))
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(indices), self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                batch_paths = df.iloc[batch_indices]['face_path'].tolist()
                batch_labels = df.iloc[batch_indices]['label'].tolist()
                
                batch_images = np.array([self.load_face_image(path) for path in batch_paths])
                batch_labels = np.array(batch_labels)
                
                yield batch_images, batch_labels
        
        return generator
    
    def build_model(self, model_type='efficientnet'):
        """
        Build and compile the deepfake detection model.
        
        Args:
            model_type (str): Type of model to build (simple, mobilenet, efficientnet)
            
        Returns:
            Model: Compiled model
        """
        if model_type == 'simple':
            # Build a simple CNN model
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])
        
        elif model_type == 'mobilenet':
            # Use MobileNetV2 as base model for transfer learning
            base_model = MobileNetV2(
                input_shape=(*self.image_size, 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Freeze the base model
            base_model.trainable = False
            
            # Add classification head
            inputs = Input(shape=(*self.image_size, 3))
            x = base_model(inputs, training=False)
            x = GlobalAveragePooling2D()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            
            model = Model(inputs, outputs)
            
        elif model_type == 'efficientnet':
            # Use EfficientNetB0 as base model for transfer learning
            base_model = EfficientNetB0(
                input_shape=(*self.image_size, 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Freeze the base model
            base_model.trainable = False
            
            # Add classification head
            inputs = Input(shape=(*self.image_size, 3))
            x = base_model(inputs, training=False)
            x = GlobalAveragePooling2D()(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.3)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            
            model = Model(inputs, outputs)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        self.model = model
        return model
    
    def train_model(self, train_generator, val_generator, epochs=20, steps_per_epoch=None, validation_steps=None):
        """
        Train the deepfake detection model.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs (int): Number of training epochs
            steps_per_epoch (int): Number of steps per epoch
            validation_steps (int): Number of validation steps
            
        Returns:
            History: Training history
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() first.")
        
        # Create model checkpoint callback
        checkpoint_dir = os.path.join(self.base_path, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'model_best.h5')
        
        callbacks = [
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                mode='max'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                verbose=1,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                verbose=1,
                min_lr=1e-6
            )
        ]
        
        # Create TensorFlow dataset from generators
        train_dataset = tf.data.Dataset.from_generator(
            train_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, *self.image_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_generator(
            val_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, *self.image_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)
        
        # Train the model
        self.history = self.model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on the test set.
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Make predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'])
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake']
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save the confusion matrix
        results_dir = os.path.join(self.base_path, 'results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Plot training history if available
        if self.history:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'])
            plt.plot(self.history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(['Train', 'Validation'], loc='lower right')
            
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Train', 'Validation'], loc='upper right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'training_history.png'))
            plt.close()
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def save_model(self, model_path=None):
        """
        Save the trained model.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if model_path is None:
            model_path = os.path.join(self.base_path, 'models', 'deepfake_detector.h5')
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return self.model
    
    def predict_video(self, video_path, output_path=None):
        """
        Predict whether a video is real or fake.
        
        Args:
            video_path (str): Path to the video
            output_path (str): Path to save the annotated video
            
        Returns:
            float: Probability of the video being fake
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train a model first.")
        
        # Create temporary directory for face extraction
        temp_dir = os.path.join(self.base_path, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Extract faces from the video
        face_paths = self.extract_faces_from_video(video_path, temp_dir)
        
        if not face_paths:
            print(f"No faces detected in the video: {video_path}")
            return None
        
        # Load and preprocess the face images
        face_images = np.array([self.load_face_image(path) for path in face_paths])
        
        # Make predictions
        predictions = self.model.predict(face_images)
        
        # Calculate the average prediction
        avg_prediction = np.mean(predictions)
        
        print(f"Prediction for {video_path}:")
        print(f"  Probability of being fake: {avg_prediction:.4f}")
        print(f"  Classification: {'Fake' if avg_prediction > 0.5 else 'Real'}")
        
        # If output path is specified, create an annotated video
        if output_path:
            self._create_annotated_video(video_path, output_path, predictions)
        
        return avg_prediction
    
    def _create_annotated_video(self, input_path, output_path, face_predictions):
        """
        Create an annotated video with deepfake detection results.
        
        Args:
            input_path (str): Path to the input video
            output_path (str): Path to save the annotated video
            face_predictions (list): Predictions for each detected face
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error opening video file: {input_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        face_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            # Process each face in the frame
            for (x, y, w, h) in faces:
                if face_idx < len(face_predictions):
                    # Get prediction for this face
                    pred = face_predictions[face_idx][0]
                    face_idx += 1
                    
                    # Determine color based on prediction
                    # Red for fake, green for real
                    color = (0, 0, 255) if pred > 0.5 else (0, 255, 0)
                    
                    # Draw rectangle around the face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    # Add prediction text
                    text = f"Fake: {pred:.2f}"
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Write the frame to the output video
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        print(f"Annotated video saved to {output_path}")


def main():
    """
    Main function to run the deepfake detection pipeline.
    """
    # Set path to the UADFV dataset
    base_path = r"C:\Users\Asus\Downloads\archive (1)\UADFV"
    
    # Initialize the deepfake detector
    detector = DeepfakeDetector(base_path)
    
    # Process the dataset
    print("Loading metadata...")
    metadata_df = detector.load_metadata()
    
    print("Preprocessing dataset...")
    faces_dir = os.path.join(base_path, 'faces')
    face_df = detector.preprocess_dataset(metadata_df, faces_dir)
    
    # Create data generators
    print("Creating data generators...")
    train_generator, val_generator, X_test, y_test = detector.create_data_generators(face_df)
    
    # Build the model
    print("Building model...")
    detector.build_model(model_type='efficientnet')
    
    # Train the model
    print("Training model...")
    detector.train_model(
        train_generator,
        val_generator,
        epochs=20,
        steps_per_epoch=len(face_df) // (2 * detector.batch_size),
        validation_steps=len(face_df) // (6 * detector.batch_size)
    )
    
    # Evaluate the model
    print("Evaluating model...")
    evaluation = detector.evaluate_model(X_test, y_test)
    
    # Save the model
    print("Saving model...")
    detector.save_model()
    
    print("Deepfake detection pipeline completed!")


if __name__ == "__main__":
    main()