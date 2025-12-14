import cv2
import torch
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO
from plyer import notification
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import time

# Load YOLO model
model = YOLO("yolov8n.pt")

# Email Credentials (Replace with valid details)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "*****@gmail.com"
EMAIL_PASSWORD = "******"
RECIPIENT_EMAIL = "*****@gmail.com"

# Image Paths (Replace with your image paths)
IMAGE_FILES = [f"network/images/image{i}.jpg" for i in range(1, 9)]

# Function to send email alert
def send_email_alert(head_count):
    subject = "Head Count Alert!"
    body = f"Warning! The detected head count has exceeded the threshold. Detected heads: {head_count}."
    
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to load and resize image
def load_image(image_path, resize_dim=(640, 480)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, resize_dim)
    return image

# Function to detect heads
def detect_heads(image_path, threshold=5):
    image = load_image(image_path)
    results = model(image)[0]
    
    head_count = sum(1 for detection in results.boxes.data if int(detection[5]) == 0 and detection[4] > 0.5)
    
    for detection in results.boxes.data:
        x1, y1, x2, y2, conf, class_id = detection.tolist()
        if int(class_id) == 0 and conf > 0.5:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, "Head", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image, head_count

# GUI Setup
def create_gui():
    root = tk.Tk()
    root.title("Real-Time Headcount Detection & Alert System")
    root.geometry("1280x800")
    default_bg = "#F2F7FB"
    flash_alert = "#FFE6E6"
    flash_safe = "#E6FFEA"

    root.configure(bg=default_bg)

    # --- Title ---
    title_label = tk.Label(
        root,
        text="🧠 Real-Time Headcount Detection & Alert System",
        font=("Times New Roman", 26, "bold"),
        bg="#0B3C5D",
        fg="white",
        pady=12
    )
    title_label.pack(fill=tk.X)

    # --- Main layout: thumbnails left, processed image right ---
    main_frame = tk.Frame(root, bg=default_bg)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    # --- Thumbnail section (Left column) ---
    thumbnail_frame = tk.LabelFrame(
        main_frame,
        text="📷 Select an Image",
        font=("Times New Roman", 16, "bold"),
        bg=default_bg,
        fg="#003366",
        padx=10,
        pady=10,
        relief="groove",
        bd=3
    )
    thumbnail_frame.pack(side=tk.LEFT, padx=10, pady=10)

    # Arrange thumbnails in 2 columns × 4 rows (Vertical block)
    for index, img_path in enumerate(IMAGE_FILES):
        img = Image.open(img_path).resize((100, 100))
        tk_img = ImageTk.PhotoImage(img)

        btn = tk.Button(
            thumbnail_frame,
            image=tk_img,
            command=lambda p=img_path: analyze_image(p),
            bd=2,
            relief="raised",
            bg="#FFFFFF",
            activebackground="#D6EAF8",
            cursor="hand2"
        )
        btn.image = tk_img
        row = index % 4
        col = index // 4
        btn.grid(row=row, column=col, padx=10, pady=10)

    # --- Output frame (Right side) ---
    output_frame = tk.LabelFrame(
        main_frame,
        text="🎯 Processed Output",
        font=("Times New Roman", 16, "bold"),
        bg="#FFFFFF",
        fg="#003366",
        padx=20,
        pady=20,
        relief="ridge",
        bd=4
    )
    output_frame.pack(side=tk.RIGHT, padx=20, pady=10, expand=True, fill=tk.BOTH)

    img_label = tk.Label(output_frame, bg="#FFFFFF")
    img_label.pack(expand=True)

    # --- Analyze Image Function ---
    def analyze_image(img_path):
        processed_image, head_count = detect_heads(img_path)

        resized_img = cv2.resize(processed_image, (600, 400))
        img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        img_label.configure(image=img)
        img_label.image = img

        # Flash Background
        for _ in range(2):
            flash_color = flash_alert if head_count > 5 else flash_safe
            root.configure(bg=flash_color)
            main_frame.configure(bg=flash_color)
            output_frame.configure(bg=flash_color)
            thumbnail_frame.configure(bg=flash_color)
            root.update()
            time.sleep(0.3)
            root.configure(bg=default_bg)
            main_frame.configure(bg=default_bg)
            output_frame.configure(bg="#FFFFFF")
            thumbnail_frame.configure(bg=default_bg)
            root.update()
            time.sleep(0.3)

        # Show Alerts
        if head_count > 5:
            notification.notify(
                title="🚨 Head Count Alert!",
                message=f"{head_count} heads detected. Threshold exceeded!",
                timeout=5
            )
            send_email_alert(head_count)
            messagebox.showwarning("Warning", f"⚠️ Head count exceeded: {head_count}")
        else:
            messagebox.showinfo("Info", f"✅ Head count is normal: {head_count}")

    # --- Footer ---
    footer = tk.Label(
        root,
        text="🔧 Powered by YOLOv8 | Built with  using OpenCV + Tkinter",
        font=("Times New Roman", 11, "italic"),
        bg="#0B3C5D",
        fg="white",
        pady=6
    )
    footer.pack(fill=tk.X, side=tk.BOTTOM)

    root.mainloop()


# Run GUI
create_gui()
