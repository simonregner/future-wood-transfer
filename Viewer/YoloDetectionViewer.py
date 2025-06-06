import os
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO


class YOLOMaskEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Mask Editor")

        self.txt_files = []
        self.current_file_index = 0
        self.current_segmentation_index = 0
        self.segments = []

        self.canvas = tk.Canvas(root, bg='gray')
        self.canvas_scale = 1.0
        self.canvas.pack(fill=tk.BOTH, expand=True)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.load_btn = tk.Button(btn_frame, text="Load TXT Folder", command=self.load_folder)
        self.load_btn.grid(row=0, column=0, padx=5)

        self.prev_btn = tk.Button(btn_frame, text="Previous", command=self.prev_segment, state='disabled')
        self.prev_btn.grid(row=0, column=1, padx=5)

        self.next_btn = tk.Button(btn_frame, text="Next", command=self.next_segment, state='disabled')
        self.next_btn.grid(row=0, column=2, padx=5)

        self.skip_lane_var = tk.IntVar()
        self.skip_lane_check = tk.Checkbutton(btn_frame, text="Skip lane masks (class 7)", variable=self.skip_lane_var,
                                              command=self.toggle_skip_lane)
        self.skip_lane_check.grid(row=0, column=3, padx=5)

        self.detect_btn = tk.Button(btn_frame, text="Run YOLO on Folder", command=self.run_yolo_on_folder)
        self.detect_btn.grid(row=0, column=4, padx=5)



        self.status_label = tk.Label(root, text="No folder loaded.")
        self.class_summary = tk.Label(root, text="", anchor='center', justify='center')

        self.status_label.pack()
        self.class_summary.pack()
        self.class_summary.pack()

        self.save_info = tk.Label(root, text="", anchor='e', justify='right')
        self.save_info.place(x=540, y=10)

        self.canvas.bind("<B1-Motion>", self.freehand_remove)
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-s>", self.save_current_mask_to_txt)
        self.root.bind("<Control-Return>", self.force_continue)
        self.root.bind("<Control-Right>", lambda e: self.next_segment())
        self.root.bind("<Control-Left>", lambda e: self.prev_segment())

        self.history = []
        self.original_img = None

    def run_yolo_on_folder(self):
        image_folder = filedialog.askdirectory(title="Select Folder with Images")
        if not image_folder:
            return

        save_img_folder = os.path.join(image_folder, "images")
        save_label_folder = os.path.join(image_folder, "labels")
        os.makedirs(save_img_folder, exist_ok=True)
        os.makedirs(save_label_folder, exist_ok=True)

        model = YOLO("best.pt")  # Replace with your model path
        extensions = [".png", ".jpg", ".jpeg", ".bmp"]

        for file in os.listdir(image_folder):
            if not any(file.lower().endswith(ext) for ext in extensions):
                continue
            img_path = os.path.join(image_folder, file)
            results = model(img_path, conf=0.7, retina_masks=True, classes=[2, 3, 4, 5])[0]

            if results.masks is None or results.masks.data is None:
                continue

            img_name = os.path.splitext(file)[0]
            label_path = os.path.join(save_label_folder, img_name + ".txt")
            output_lines = []

            for i, mask_tensor in enumerate(results.masks.data):
                mask_array = mask_tensor.cpu().numpy()
                mask_img = (mask_array * 255).astype(np.uint8)

                contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                largest_contour = max(contours, key=cv2.contourArea)
                if largest_contour.shape[0] < 3:
                    continue
                height, width = mask_img.shape[:2]
                norm_points = [f"{pt[0][0] / width:.6f} {pt[0][1] / height:.6f}" for pt in largest_contour]
                if results.boxes and results.boxes.cls is not None:
                    class_id = int(results.boxes.cls[i].item())
                else:
                    continue  # fallback if class is not available
                output_lines.append(f"{class_id} {' '.join(norm_points)}")

            print(len(output_lines))

            if output_lines:
                with open(label_path, 'w') as f:
                    f.write("\n" + "\n".join(output_lines))
                new_img_path = os.path.join(save_img_folder, file)
                cv2.imwrite(new_img_path, cv2.imread(img_path))

        self.status_label.config(text="YOLO detection completed. Files saved.")

    def load_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder with YOLO txt files")
        if folder_path:
            self.txt_files = []
            for root_dir, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.txt') and file != 'classes.txt':
                        self.txt_files.append(os.path.join(root_dir, file))
            self.txt_files.sort()
            if self.txt_files:
                self.current_file_index = 0
                self.load_segmentations()

    def load_segmentations(self):
        while self.current_file_index < len(self.txt_files):
            with open(self.txt_files[self.current_file_index], 'r', encoding='utf-8', errors='ignore') as f:
                if any(line.strip() for line in f):
                    break
                else:
                    self.current_file_index += 1
        if self.current_file_index >= len(self.txt_files):
            self.status_label.config(text="No non-empty TXT files found.")
            return
        self.segments = []
        class_counts = [0] * 8

        with open(self.txt_files[self.current_file_index], 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
        for line in lines:
            if line.strip():
                try:
                    class_id = int(line.split()[0])
                except (ValueError, IndexError):
                    continue
                class_counts[class_id] += 1
                self.segments.append((class_id, self.parse_yolo_line(line.strip())))

        self.current_segmentation_index = 0
        if not self.segments:
            self.status_label.config(text="No segments found.")
            self.rgb_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            self.mask_img = np.zeros_like(self.rgb_img, dtype=np.uint8)
            self.original_img = self.mask_img.copy()
            self.refresh_canvas()
            return

        self.show_segmentation()
        class_names = [
            "background", "bugle_road", "left_turn", "right_turn",
            "road", "straight_turn", "intersection", "lane"
        ]
        class_info = "  |  ".join(
            f"{name}: {count}" if name != "lane" else f"lane: {count} ✅"
            for name, count in zip(class_names, class_counts) if count > 0
        )
        self.class_summary.config(text=f"Classes in file: {class_info}")
        self.update_buttons()

    def parse_yolo_line(self, line):
        parts = line.split()
        points = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts), 2)]
        return points

    def show_segmentation(self):
        self.save_info.config(text="")

        txt_file = self.txt_files[self.current_file_index]
        image_dir = os.path.join(os.path.dirname(os.path.dirname(txt_file)), "images")
        base_name = os.path.splitext(os.path.basename(txt_file))[0]

        extensions = [".png", ".jpg", ".jpeg", ".bmp"]
        rgb_image = None
        for ext in extensions:
            candidate = os.path.join(image_dir, base_name + ext)
            if os.path.exists(candidate):
                rgb_image = cv2.imread(candidate.strip())
                break

        if rgb_image is not None:
            scale = 1080 / rgb_image.shape[0]
            new_width = int(rgb_image.shape[1] * scale)
            self.rgb_img = cv2.resize(rgb_image, (new_width, 1080))
        else:
            self.rgb_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.mask_img = np.zeros_like(self.rgb_img, dtype=np.uint8)
        self.rgb_image_shape = self.rgb_img.shape[:2]

        class_id, points = self.segments[self.current_segmentation_index]
        height, width = self.rgb_image_shape
        scaled_pts = np.array([[int(x * width), int(y * height)] for x, y in points], np.int32).reshape((-1, 1, 2))

        if class_id == 7:
            cv2.fillPoly(self.mask_img, [scaled_pts], (255, 0, 0))
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [scaled_pts], 255)
            kernel = np.ones((10, 10), np.uint8)
            eroded = cv2.erode(mask, kernel, iterations=1)
            border = cv2.subtract(mask, eroded)
            self.mask_img[border > 0] = (255, 0, 0)

        cv2.rectangle(self.mask_img, (0, height - 20), (width, height), (0, 0, 0), -1)
        cv2.rectangle(self.mask_img, (0, 0), (20, height), (0, 0, 0), -1)
        cv2.rectangle(self.mask_img, (width - 20, 0), (width, height), (0, 0, 0), -1)
        self.original_img = self.mask_img.copy()
        self.refresh_canvas()

    def refresh_canvas(self):
        self.canvas.update_idletasks()
        scaled_width = int(self.rgb_img.shape[1] * self.canvas_scale)
        scaled_height = int(self.rgb_img.shape[0] * self.canvas_scale)
        rgb_resized = cv2.resize(self.rgb_img, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)
        mask_resized = cv2.resize(self.mask_img, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)
        mask_binary = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY) > 0
        img_resized = rgb_resized.copy()
        img_resized[mask_binary] = mask_resized[mask_binary]
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        x_offset = max((canvas_width - scaled_width) // 2, 0)
        y_offset = max((canvas_height - scaled_height) // 2, 0)
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
        self.canvas.img_tk = img_tk
    def prev_segment(self):
        if not np.array_equal(self.mask_img, self.original_img):
            self.save_info.config(
                text="Unsaved changes! Press Ctrl+S to save or Ctrl+Enter to continue without saving.", fg="red")
            self._pending_action = 'prev'
            return
        if self.current_segmentation_index > 0:
            self.current_segmentation_index -= 1
        elif self.current_file_index > 0:
            self.current_file_index -= 1
            self.load_segmentations()
            self.current_segmentation_index = len(self.segments) - 1
        self.show_segmentation()
        self.update_buttons()

    def next_segment(self):
        skip_class_7 = getattr(self, 'skip_lane_enabled', False)
        while skip_class_7 and self.current_file_index < len(self.txt_files):
            with open(self.txt_files[self.current_file_index], 'r') as f:
                if any(line.startswith('7') for line in f):
                    self.current_file_index += 1
                else:
                    break
        if not np.array_equal(self.mask_img, self.original_img):
            self.save_info.config(
                text="Unsaved changes! Press Ctrl+S to save or Ctrl+Enter to continue without saving.", fg="red")
            self._pending_action = 'next'
            return
        if self.current_segmentation_index < len(self.segments) - 1:
            self.current_segmentation_index += 1
        elif self.current_file_index < len(self.txt_files) - 1:
            self.current_file_index += 1
            self.load_segmentations()
        self.show_segmentation()
        self.update_buttons()

    def freehand_remove(self, event):
        self.canvas.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        scaled_width = int(self.rgb_img.shape[1] * self.canvas_scale)
        scaled_height = int(self.rgb_img.shape[0] * self.canvas_scale)
        x_offset = max((canvas_width - scaled_width) // 2, 0)
        y_offset = max((canvas_height - scaled_height) // 2, 0)

        x = int((event.x - x_offset) / self.canvas_scale)
        y = int((event.y - y_offset) / self.canvas_scale)

        if x < 0 or y < 0 or x >= self.mask_img.shape[1] or y >= self.mask_img.shape[0]:
            return

        self.history.append(self.mask_img.copy())
        if len(self.history) > 50:
            self.history.pop(0)
        cv2.circle(self.mask_img, (x, y), 10, (0, 0, 0), -1)
        self.refresh_canvas()

    def undo(self, event=None):
        if self.history:
            self.mask_img = self.history.pop()
            self.refresh_canvas()

    def save_current_mask_to_txt(self, event=None):
        if hasattr(self, '_force_save') and self._force_save:
            self._force_save = False
        elif np.array_equal(self.mask_img, self.original_img):
            self.save_info.config(text="No changes to save.", fg="gray")
            return

        gray = cv2.cvtColor(self.mask_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 2:
            self.save_info.config(text="Only 1 polygon found. Press Ctrl+Enter to force save.", fg="red")
            self._pending_action = 'save'
            return
        elif len(contours) > 2:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        output_lines = []
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 1:
                continue
            height, width = self.rgb_image_shape
            norm_points = [f"{pt[0] / width:.6f} {pt[1] / height:.6f}" for pt in contour]
            line = f"7 {' '.join(norm_points)}"
            output_lines.append(line)

        with open(self.txt_files[self.current_file_index], 'a') as f:
            f.write(" " + " ".join(output_lines))
        self.original_img = self.mask_img.copy()
        self.save_info.config(text="Changes saved.", fg="green")

    def force_continue(self, event=None):
        if hasattr(self, '_pending_action'):
            if self._pending_action == 'save':
                self._force_save = True
                self._pending_action = None
                self.save_current_mask_to_txt()
                return
            if self._pending_action == 'next':
                self._pending_action = None
                if self.current_segmentation_index < len(self.segments) - 1:
                    self.current_segmentation_index += 1
                elif self.current_file_index < len(self.txt_files) - 1:
                    self.current_file_index += 1
                    self.load_segmentations()
                self.show_segmentation()
                self.update_buttons()
            elif self._pending_action == 'prev':
                self._pending_action = None
                if self.current_segmentation_index > 0:
                    self.current_segmentation_index -= 1
                elif self.current_file_index > 0:
                    self.current_file_index -= 1
                    self.load_segmentations()
                    self.current_segmentation_index = len(self.segments) - 1
                self.show_segmentation()
                self.update_buttons()

    def toggle_skip_lane(self):
        self.skip_lane_enabled = bool(self.skip_lane_var.get())

    def update_buttons(self):
        self.prev_btn.config(
            state='normal' if (self.current_segmentation_index > 0 or self.current_file_index > 0) else 'disabled')
        self.next_btn.config(state='normal' if (
            self.current_segmentation_index < len(self.segments) - 1 or self.current_file_index < len(
                self.txt_files) - 1) else 'disabled')

    def delete_current_image(self):
        if not self.txt_files:
            return

        txt_file = self.txt_files[self.current_file_index]
        img_folder = os.path.join(os.path.dirname(os.path.dirname(txt_file)), "images")
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        extensions = [".png", ".jpg", ".jpeg", ".bmp"]

        os.remove(txt_file)
        for ext in extensions:
            img_path = os.path.join(img_folder, base_name + ext)
            if os.path.exists(img_path):
                os.remove(img_path)
                break

        self.txt_files.pop(self.current_file_index)
        if self.current_file_index >= len(self.txt_files):
            self.current_file_index = max(0, len(self.txt_files) - 1)

        if self.txt_files:
            self.load_segmentations()
        else:
            self.status_label.config(text="No files left.")
            self.canvas.delete("all")
            self.class_summary.config(text="")

        self.prev_btn.config(
            state='normal' if (self.current_segmentation_index > 0 or self.current_file_index > 0) else 'disabled')
        self.next_btn.config(state='normal' if (
                self.current_segmentation_index < len(self.segments) - 1 or self.current_file_index < len(
            self.txt_files) - 1) else 'disabled')


if __name__ == "__main__":
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    target_width = 2200
    target_height = 1300

    if screen_width < target_width or screen_height < target_height:
        target_width = min(target_width, screen_width)
        target_height = min(target_height, screen_height)

    root.geometry(f'{target_width}x{target_height}')
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    app = YOLOMaskEditor(root)
    root.mainloop()
