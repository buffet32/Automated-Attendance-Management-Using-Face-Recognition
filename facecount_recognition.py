import cv2
import numpy as np
import os
import json
import pickle
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import threading
from retinaface import RetinaFace
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

class FaceRecognitionSystem:
    def __init__(self):
        """Initialize face detection and recognition models"""
        # RetinaFace for detection
        self.detection_model = 'retinaface'
        
        # FaceNet for embeddings
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(
            keep_all=True,
            min_face_size=60,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            device=self.device
        )
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Face database
        self.face_database = {}
        self.database_file = 'face_database.pkl'
        self.load_database()
        
    def detect_faces(self, image):
        """Detect faces using RetinaFace"""
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            faces = RetinaFace.detect_faces(image_rgb)
            
            face_boxes = []
            if isinstance(faces, dict):
                for key in faces.keys():
                    face = faces[key]
                    facial_area = face['facial_area']
                    x1, y1, x2, y2 = facial_area
                    face_boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    
            return face_boxes
            
        except Exception as e:
            print(f"RetinaFace detection error: {e}")
            return []
    
    def extract_face_embedding(self, image, face_box):
        """Extract face embedding using FaceNet"""
        try:
            x1, y1, x2, y2 = face_box
            
            # Crop face from image
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
                
            # Convert to RGB if needed
            if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_crop
            
            # Convert to PIL Image
            face_pil = Image.fromarray(face_rgb)
            
            # Use MTCNN to get aligned face
            face_tensor = self.mtcnn(face_pil)
            
            if face_tensor is None:
                return None
                
            # MTCNN returns a tensor for the detected face, which is already correctly shaped.
            # We just need to ensure it's on the correct device.
            face_tensor = face_tensor.to(self.device)
            
            with torch.no_grad():
                embedding = self.facenet(face_tensor)
                embedding = F.normalize(embedding, p=2, dim=1)
                
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Embedding extraction error: {e}")
            return None
    
    def add_person_to_database(self, name, image, face_box):
        """Add a person's face embedding to the database"""
        embedding = self.extract_face_embedding(image, face_box)
        
        if embedding is not None:
            if name not in self.face_database:
                self.face_database[name] = []
            
            self.face_database[name].append(embedding)
            self.save_database()
            return True
        return False
    
    def recognize_face(self, image, face_box, threshold=0.6):
        """Recognize a face by comparing with database"""
        embedding = self.extract_face_embedding(image, face_box)
        
        if embedding is None:
            return "Unknown", 0.0
        
        best_match = "Unknown"
        best_similarity = 0.0
        
        for name, embeddings in self.face_database.items():
            for stored_embedding in embeddings:
                # Calculate cosine similarity
                similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
                
                if similarity > best_similarity and similarity > threshold:
                    best_similarity = similarity
                    best_match = name
        
        return best_match, best_similarity
    
    def save_database(self):
        """Save face database to file"""
        try:
            with open(self.database_file, 'wb') as f:
                pickle.dump(self.face_database, f)
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def load_database(self):
        """Load face database from file"""
        try:
            if os.path.exists(self.database_file):
                with open(self.database_file, 'rb') as f:
                    self.face_database = pickle.load(f)
            else:
                self.face_database = {}
        except Exception as e:
            print(f"Error loading database: {e}")
            self.face_database = {}

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FaceCount - Recognition System")
        
        # Initialize recognition system
        self.face_system = FaceRecognitionSystem()
        
        # Create GUI
        self.setup_gui()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="FaceCount - Recognition System",
            font=('Arial', 12)
        )
        title_label.pack(pady=5)
        
        # Image display
        self.image_label = ttk.Label(main_frame)
        self.image_label.pack(pady=10)
        
        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Buttons
        ttk.Button(
            controls_frame, 
            text="Open Image", 
            command=self.load_image
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            controls_frame, 
            text="Take Photo", 
            command=self.take_photo
        ).pack(side=tk.LEFT, padx=5)
        
        self.detect_btn = ttk.Button(
            controls_frame, 
            text="Detect & Recognize", 
            state=tk.DISABLED,
            command=self.detect_and_recognize
        )
        self.detect_btn.pack(side=tk.LEFT, padx=5)
        
        # Database management buttons
        db_frame = ttk.Frame(main_frame)
        db_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            db_frame, 
            text="Add Person", 
            command=self.add_person_mode
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            db_frame, 
            text="View Database", 
            command=self.view_database
        ).pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(
            db_frame, 
            text="Save Attendance", 
            state=tk.DISABLED,
            command=self.save_attendance
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Class selection and info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(info_frame, text="Class:").pack(side=tk.LEFT, padx=5)
        
        self.class_var = tk.StringVar(value="3IIR")
        class_combo = ttk.Combobox(
            info_frame, 
            textvariable=self.class_var,
            values=["3IIR", "4IIR", "5IIR"],
            state="readonly",
            width=10
        )
        class_combo.pack(side=tk.LEFT, padx=5)
        
        # Recognition results
        ttk.Label(info_frame, text="Recognized:").pack(side=tk.LEFT, padx=(20, 5))
        self.recognized_count_var = tk.StringVar(value="0")
        ttk.Label(
            info_frame, 
            textvariable=self.recognized_count_var,
            font=('Arial', 11)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(info_frame, text="Unknown:").pack(side=tk.LEFT, padx=(10, 5))
        self.unknown_count_var = tk.StringVar(value="0")
        ttk.Label(
            info_frame, 
            textvariable=self.unknown_count_var,
            font=('Arial', 11)
        ).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            main_frame, 
            textvariable=self.status_var,
            font=('Arial', 9),
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        # Initialize variables
        self.current_image = None
        self.faces = []
        self.recognition_results = []
        self.add_person_mode_active = False
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.process_image(file_path)
    
    def take_photo(self):
        self.status_var.set("Preparing camera...")
        self.root.update()
        
        threading.Thread(target=self._capture_photo, daemon=True).start()
    
    def _capture_photo(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.root.after(0, lambda: self.status_var.set("Error: Could not open camera"))
            return
            
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            temp_path = "temp_capture_recognition.jpg"
            cv2.imwrite(temp_path, frame)
            self.root.after(0, lambda: self.process_image(temp_path))
        else:
            self.root.after(0, lambda: self.status_var.set("Failed to capture image"))
    
    def process_image(self, image_path):
        try:
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                raise ValueError("Could not read the image")
                
            img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            img_rgb = self.resize_image(img_rgb, 800, 600)
            
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
            
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo
            
            # Reset counters
            self.recognized_count_var.set("0")
            self.unknown_count_var.set("0")
            self.faces = []
            self.recognition_results = []
            
            self.detect_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.DISABLED)
            self.status_var.set(f"Loaded: {os.path.basename(image_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            self.status_var.set("Error processing image")
    
    def detect_and_recognize(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        self.status_var.set("Detecting and recognizing faces...")
        self.root.update()
        
        threading.Thread(target=self._detect_and_recognize_thread, daemon=True).start()
    
    def _detect_and_recognize_thread(self):
        try:
            img_copy = self.current_image.copy()
            
            # Detect faces
            self.faces = self.face_system.detect_faces(img_copy)
            
            # Recognize each face
            self.recognition_results = []
            recognized_count = 0
            unknown_count = 0
            
            img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            
            for i, face_box in enumerate(self.faces):
                x1, y1, x2, y2 = face_box
                
                # Recognize face
                name, confidence = self.face_system.recognize_face(img_copy, face_box)
                self.recognition_results.append((name, confidence))
                
                if name != "Unknown":
                    recognized_count += 1
                    color = (0, 255, 0)  # Green for recognized
                    label = f'{name} ({confidence:.2f})'
                else:
                    unknown_count += 1
                    color = (255, 165, 0)  # Orange for unknown
                    label = f'Unknown {unknown_count}'
                
                # Draw rectangle and label
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 3)
                
                # Add label with background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(img_rgb, (x1, y1-30), (x1 + label_size[0], y1), color, -1)
                cv2.putText(img_rgb, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Update display in main thread
            self.root.after(0, self._update_recognition_display, img_rgb, recognized_count, unknown_count)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Recognition error: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Error during recognition"))
    
    def _update_recognition_display(self, img_rgb, recognized_count, unknown_count):
        try:
            img_rgb = self.resize_image(img_rgb, 800, 600)
            
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
            
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo
            
            # Update counters
            self.recognized_count_var.set(str(recognized_count))
            self.unknown_count_var.set(str(unknown_count))
            
            self.save_btn.config(state=tk.NORMAL)
            self.status_var.set(f"Recognition complete: {recognized_count} known, {unknown_count} unknown")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update display: {str(e)}")
    
    def add_person_mode(self):
        if self.current_image is None or len(self.faces) == 0:
            messagebox.showwarning("Warning", "Please detect faces first")
            return
        
        # Show dialog to select face and enter name
        self.show_add_person_dialog()
    
    def show_add_person_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Person to Database")
        dialog.geometry("400x300")
        
        ttk.Label(dialog, text="Select a face to add:", font=('Arial', 11)).pack(pady=10)
        
        # Create buttons for each detected face
        for i, face_box in enumerate(self.faces):
            x1, y1, x2, y2 = face_box
            
            # Extract face image for preview
            face_crop = self.current_image[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (80, 80))
            
            face_img = ImageTk.PhotoImage(image=Image.fromarray(face_resized))
            
            btn = ttk.Button(
                dialog, 
                text=f"Face {i+1}",
                command=lambda idx=i: self.add_selected_face(dialog, idx)
            )
            btn.pack(pady=5)
    
    def add_selected_face(self, dialog, face_index):
        dialog.destroy()
        
        # Get person name
        name = simpledialog.askstring("Add Person", "Enter person's name:")
        
        if name and name.strip():
            name = name.strip()
            face_box = self.faces[face_index]
            
            if self.face_system.add_person_to_database(name, self.current_image, face_box):
                messagebox.showinfo("Success", f"Added {name} to database")
                self.status_var.set(f"Added {name} to database")
            else:
                messagebox.showerror("Error", "Failed to add person to database")
    
    def view_database(self):
        if not self.face_system.face_database:
            messagebox.showinfo("Database", "Database is empty")
            return
        
        # Create database viewer window
        db_window = tk.Toplevel(self.root)
        db_window.title("Face Database")
        db_window.geometry("300x400")
        
        ttk.Label(db_window, text="Known People:", font=('Arial', 12)).pack(pady=10)
        
        # Create listbox with scrollbar
        frame = ttk.Frame(db_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        listbox = tk.Listbox(frame)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        
        for name, embeddings in self.face_system.face_database.items():
            listbox.insert(tk.END, f"{name} ({len(embeddings)} samples)")
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def save_attendance(self):
        if len(self.faces) == 0:
            messagebox.showwarning("Warning", "No faces detected to save")
            return
            
        try:
            os.makedirs('attendance', exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            class_name = self.class_var.get()
            
            recognized_count = int(self.recognized_count_var.get())
            unknown_count = int(self.unknown_count_var.get())
            
            # Save detailed attendance log
            with open('attendance/attendance_log.txt', 'a') as f:
                f.write(f"{timestamp} | Class: {class_name} | Recognized: {recognized_count} | Unknown: {unknown_count} | Method: Recognition\n")
                
                # Add individual recognition results
                for i, (name, confidence) in enumerate(self.recognition_results):
                    f.write(f"  Face {i+1}: {name} (confidence: {confidence:.3f})\n")
            
            # Save the image with recognition results
            img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            for i, (face_box, (name, confidence)) in enumerate(zip(self.faces, self.recognition_results)):
                x1, y1, x2, y2 = face_box
                
                if name != "Unknown":
                    color = (0, 255, 0)
                    label = f'{name} ({confidence:.2f})'
                else:
                    color = (255, 165, 0)
                    label = f'Unknown {i+1}'
                
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 3)
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(img_rgb, (x1, y1-30), (x1 + label_size[0], y1), color, -1)
                cv2.putText(img_rgb, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            image_path = f'attendance/attendance_{class_name}_{timestamp}_recognition.jpg'
            cv2.imwrite(image_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            
            self.status_var.set(f"Saved: {recognized_count} recognized, {unknown_count} unknown")
            messagebox.showinfo("Success", f"Attendance saved for {class_name}\nRecognized: {recognized_count}, Unknown: {unknown_count}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save attendance: {str(e)}")
    
    @staticmethod
    def resize_image(image, max_width, max_height):
        h, w = image.shape[:2]
        
        if h <= max_height and w <= max_width:
            return image
            
        ratio = min(max_width/w, max_height/h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def main():
    root = tk.Tk()
    root.title("FaceCount - Recognition System")
    
    window_width = 1000
    window_height = 900
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    style = ttk.Style()
    style.configure('TButton', font=('Arial', 9))
    style.configure('TLabel', font=('Arial', 9))
    
    app = FaceRecognitionApp(root)
    
    root.mainloop()

if __name__ == "__main__":
    main()
