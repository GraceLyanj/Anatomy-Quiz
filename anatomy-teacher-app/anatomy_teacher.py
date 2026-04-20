import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import torch
from segment_anything import sam_model_registry, SamPredictor


class AnatomyTeachingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MRI Anatomy Learning App")
        self.root.geometry("950x850")

        # Create a canvas with scrollbar for the entire app
        main_canvas = tk.Canvas(root)
        scrollbar = tk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
        self.scrollable_frame = tk.Frame(main_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )

        main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)

        main_canvas.pack(side="left", fill="both", expand=True, padx=0)
        scrollbar.pack(side="right", fill="y", padx=0)

        # Load SAM model
        print("Loading segmentation model...")
        device = torch.device("cpu")
        sam = sam_model_registry["vit_b"](checkpoint="ckpt/sam_vit_b_01ec64.pth")
        sam.to(device)
        self.predictor = SamPredictor(sam)
        print("✅ Model loaded!")

        # App state
        self.current_image = None
        self.current_image_display = None
        self.user_mask = None
        self.score = 0
        self.photo = None

        # Current question
        self.current_question = {
            'image_path': 'demo_images/lung_005_160.png',
            'question': 'Click on the HEART/MEDIASTINUM area',
            'structure': 'heart'
        }

        # Setup UI
        self.setup_ui()

        # Load first question
        self.load_question()

    def setup_ui(self):
        """Create the user interface"""
        print("DEBUG: Setting up UI...")

        # Title
        title_label = tk.Label(
            self.scrollable_frame,
            text="🫀 MRI Anatomy Quiz 🧠",
            font=("Arial", 24, "bold"),
            bg="#2c3e50",
            fg="white",
            pady=10
        )
        title_label.pack(fill=tk.X)

        # Score display
        self.score_label = tk.Label(
            self.scrollable_frame,
            text=f"Score: {self.score}",
            font=("Arial", 14),
            fg="#27ae60"
        )
        self.score_label.pack(pady=5)

        # Question display
        self.question_label = tk.Label(
            self.scrollable_frame,
            text="",
            font=("Arial", 16, "bold"),
            fg="#2980b9"
        )
        self.question_label.pack(pady=10)

        # Canvas for image display
        self.canvas = tk.Canvas(
            self.scrollable_frame,
            width=512,
            height=512,
            bg='#ecf0f1',
            cursor="crosshair"
        )
        self.canvas.pack(pady=10)

        # Bind click event
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Instruction label
        self.instruction_label = tk.Label(
            self.scrollable_frame,
            text="👆 Click on the image to identify the structure",
            font=("Arial", 12),
            fg="#7f8c8d"
        )
        self.instruction_label.pack(pady=5)

        # Button frame with yellow background to see it
        btn_frame = tk.Frame(self.scrollable_frame, bg="yellow")
        btn_frame.pack(pady=15, fill=tk.X)

        print("DEBUG: Creating buttons...")

        # Submit button
        self.submit_btn = tk.Button(
            btn_frame,
            text="✓ Submit Answer",
            command=self.check_answer,
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            padx=20,
            pady=10,
            state=tk.DISABLED
        )
        self.submit_btn.pack(side=tk.LEFT, padx=10, expand=True)
        print("DEBUG: Submit button created")

        # Clear button
        clear_btn = tk.Button(
            btn_frame,
            text="↻ Clear Selection",
            command=self.clear_selection,
            font=("Arial", 12),
            bg="#95a5a6",
            fg="white",
            padx=20,
            pady=10
        )
        clear_btn.pack(side=tk.LEFT, padx=10, expand=True)
        print("DEBUG: Clear button created")

        # Next question button
        self.next_btn = tk.Button(
            btn_frame,
            text="→ Next Question",
            command=self.next_question,
            font=("Arial", 12),
            bg="#3498db",
            fg="white",
            padx=20,
            pady=10
        )
        self.next_btn.pack(side=tk.LEFT, padx=10, expand=True)
        print("DEBUG: Next button created")

        print("DEBUG: UI setup complete!")

    def load_question(self):
        """Load and display a question"""
        # Load image
        self.current_image = cv2.imread(self.current_question['image_path'])
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)

        # Set in predictor
        self.predictor.set_image(self.current_image)

        # Display question
        self.question_label.config(text=self.current_question['question'])

        # Display image
        self.display_image(self.current_image)

        # Reset state
        self.user_mask = None
        self.submit_btn.config(state=tk.DISABLED)

        print(f"✅ Question loaded: {self.current_question['question']}")

    def display_image(self, img):
        """Display image on canvas"""
        # Resize to fit canvas
        img_resized = cv2.resize(img, (512, 512))

        # Convert to PIL Image
        img_pil = Image.fromarray(img_resized)
        self.photo = ImageTk.PhotoImage(img_pil)

        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def on_canvas_click(self, event):
        """Handle user click on image"""
        x, y = event.x, event.y
        print(f"User clicked at: ({x}, {y})")

        # Convert click coordinates to original image size
        h, w = self.current_image.shape[:2]
        scale_x = w / 512
        scale_y = h / 512

        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)

        # Get segmentation
        input_point = np.array([[orig_x, orig_y]])
        input_label = np.array([1])

        print("Running segmentation...")
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        # Store the mask
        self.user_mask = masks[0]

        print(f"✅ Segmentation complete! Score: {scores[0]:.3f}")

        # Show selection
        self.show_selection(masks[0])

        # Enable submit button
        self.submit_btn.config(state=tk.NORMAL)
        print("DEBUG: Submit button enabled")

    def show_selection(self, mask):
        """Highlight the selected region"""
        # Create overlay
        overlay = self.current_image.copy()

        # Color the mask (green)
        overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5

        # Display
        self.display_image(overlay.astype(np.uint8))

    def clear_selection(self):
        """Clear the current selection"""
        self.user_mask = None
        self.display_image(self.current_image)
        self.submit_btn.config(state=tk.DISABLED)
        print("Selection cleared")

    def check_answer(self):
        """Check if the answer is correct"""
        if self.user_mask is None:
            messagebox.showwarning("No Selection", "Please click on the image first!")
            return

        # Calculate percentage of image covered
        coverage = (self.user_mask.sum() / self.user_mask.size) * 100

        print(f"DEBUG: Coverage = {coverage:.1f}%")

        # For the heart/mediastinum in this lung image:
        # - Too small (< 10%): probably clicked on a small detail
        # - Good range (10-45%): likely the heart/mediastinum area
        # - Too large (> 45%): selected too much (whole lungs, etc.)

        if 10 < coverage < 45:
            self.score += 10
            self.score_label.config(text=f"Score: {self.score}")
            messagebox.showinfo(
                "Correct! ✓",
                f"Great job! You identified the heart/mediastinum area!\n\n"
                f"Selected area: {coverage:.1f}% of image\n"
                f"Points earned: +10\n"
                f"Total score: {self.score}"
            )
        elif coverage < 10:
            messagebox.showinfo(
                "Too Small",
                f"The selected area is too small ({coverage:.1f}%).\n\n"
                f"Hint: The heart and mediastinum occupy the central chest area.\n"
                f"Try clicking more in the center of the bright area."
            )
        else:
            messagebox.showinfo(
                "Too Large",
                f"The selected area is too large ({coverage:.1f}%).\n\n"
                f"Hint: Focus on just the heart/mediastinum in the center,\n"
                f"not the entire chest or lungs."
            )

    def next_question(self):
        """Load next question"""
        messagebox.showinfo(
            "Demo Mode",
            "This is a demo with one question.\n\n"
            "In the full app, this would load the next question!"
        )
        self.clear_selection()


# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = AnatomyTeachingApp(root)
    root.mainloop()