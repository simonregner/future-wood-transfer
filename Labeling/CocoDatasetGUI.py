import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import json
import os
from PIL import Image, ImageTk


def on_mouse_press(event):
    self.drawing = True
    self.current_polygon.append((event.x, event.y))


def on_mouse_drag(event):
    if self.drawing:
        self.show_image()
        if len(self.current_polygon) > 1:
            for i in range(len(self.current_polygon) - 1):
                self.canvas.create_line(self.current_polygon[i][0], self.current_polygon[i][1],
                                        self.current_polygon[i + 1][0], self.current_polygon[i + 1][1], fill='red',
                                        width=2)
        self.canvas.create_line(self.current_polygon[-1][0], self.current_polygon[-1][1], event.x, event.y, fill='red',
                                width=2)


def on_mouse_release(event):
    if self.drawing:
        self.drawing = False
        self.current_polygon.append((event.x, event.y))
        if len(self.current_polygon) > 1:
            for i in range(len(self.current_polygon) - 1):
                self.canvas.create_line(self.current_polygon[i][0], self.current_polygon[i][1],
                                        self.current_polygon[i + 1][0], self.current_polygon[i + 1][1], fill='green',
                                        width=2)


class ImageLabelingApp:
    def __init__(self, root):
        self.root = root

        self.root.title("COCO Dataset Labeling Tool")

        self.image_folder = None
        self.image_list = []
        self.current_image_index = 0
        self.annotations = []
        self.current_polygon = []
        self.drawing = False

        self.canvas = tk.Canvas(root, width=800, height=600, bg='gray')
        self.canvas.pack()

        self.canvas.bind("<ButtonPress-1>", lambda event: self.on_mouse_press(event))
        self.canvas.bind("<B1-Motion>", lambda event: self.on_mouse_drag(event))
        self.canvas.bind("<ButtonRelease-1>", lambda event: self.on_mouse_release(event))

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()
        self.root.bind('<space>', self.finish_polygon)
        self.root.bind('<Return>', self.next_image)

        self.load_folder_button = tk.Button(self.button_frame, text="Load Folder", command=self.load_folder)
        self.load_folder_button.pack(side=tk.LEFT)

        self.load_json_button = tk.Button(self.button_frame, text="Load JSON", command=self.load_json)
        self.load_json_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(self.button_frame, text="Save Annotations", command=self.save_annotations)
        self.save_button.pack(side=tk.LEFT)

        self.prev_button = tk.Button(self.button_frame, text="Previous Image", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(self.button_frame, text="Next Image", command=self.next_image)
        self.next_button.pack(side=tk.LEFT)


def load_folder(self):
    folder_path = filedialog.askdirectory()
    if folder_path:
        self.image_folder = folder_path
        self.image_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                           f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'))]
        self.image_list.sort()
        self.current_image_index = 0
        if self.image_list:
            self.load_image()


def load_image(self):
    if self.image_list:
        self.image_path = self.image_list[self.current_image_index]
        self.image = cv2.imread(self.image_path)
        if hasattr(self, 'data'):
            self.annotations = self.get_image_annotations()
        else:
            self.annotations = []
        self.current_polygon = []
        self.show_image()
    if self.image_list:
        self.image_path = self.image_list[self.current_image_index]
        self.image = cv2.imread(self.image_path)
        self.annotations = self.get_image_annotations()
        self.current_polygon = []
        self.show_image()
    if self.image_list:
        self.image_path = self.image_list[self.current_image_index]
        self.image = cv2.imread(self.image_path)
        self.annotations = self.get_image_annotations()
    self.current_polygon = []
    self.show_image()


def show_image(self):
    self.canvas.delete("all")
    if self.image_path is not None:
        self.img_copy = self.image.copy()
        self.photo = cv2.cvtColor(self.img_copy, cv2.COLOR_BGR2RGB)
        self.photo = Image.fromarray(self.photo)
        self.photo = self.photo.resize((800, 600), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.photo)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
    self.draw_existing_annotations()


def on_mouse_press(self, event):
    self.drawing = True
    self.current_polygon.append((event.x, event.y))


def on_mouse_drag(self, event):
    if self.drawing:
        self.show_image()
        if len(self.current_polygon) > 1:
            for i in range(len(self.current_polygon) - 1):
                self.canvas.create_line(self.current_polygon[i][0], self.current_polygon[i][1],
                                        self.current_polygon[i + 1][0], self.current_polygon[i + 1][1], fill='red',
                                        width=2)
        self.canvas.create_line(self.current_polygon[-1][0], self.current_polygon[-1][1], event.x, event.y, fill='red',
                                width=2)


def on_mouse_release(self, event):
    if self.drawing:
        self.drawing = False
        self.current_polygon.append((event.x, event.y))
        if len(self.current_polygon) > 1:
            for i in range(len(self.current_polygon) - 1):
                self.canvas.create_line(self.current_polygon[i][0], self.current_polygon[i][1],
                                        self.current_polygon[i + 1][0], self.current_polygon[i + 1][1], fill='green',
                                        width=2)


def finish_polygon(self, event):
    if len(self.current_polygon) > 2:
        self.canvas.create_line(self.current_polygon[-1][0], self.current_polygon[-1][1], self.current_polygon[0][0],
                                self.current_polygon[0][1], fill='green', width=2)
        self.annotations.append({
            'segmentation': [
                [coord for point in self.current_polygon for coord in point]
            ],
            'category_id': 1  # Default category id for example
        })
        self.current_polygon = []


def load_json(self):
    json_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if json_path:
        self.json_path = json_path
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        messagebox.showinfo("Info", f"Loaded annotations from {self.json_path}")


def draw_existing_annotations(self):
    if self.annotations:
        for annotation in self.annotations:
            if 'segmentation' in annotation:
                points = annotation['segmentation'][0]
                for i in range(0, len(points) - 2, 2):
                    self.canvas.create_line(points[i], points[i + 1], points[i + 2], points[i + 3], fill='green',
                                            width=2)
                self.canvas.create_line(points[-2], points[-1], points[0], points[1], fill='green', width=2)


def get_image_annotations(self):
    if hasattr(self, 'data') and 'annotations' in self.data:
        image_name = os.path.basename(self.image_path)
        image_annotations = [ann for ann in self.data['annotations'] if
                             self.data['images'][ann['image_id'] - 1]['file_name'] == image_name]
        return image_annotations
    return []


def save_annotations(self):
    if not self.annotations:
        messagebox.showwarning("Warning", "No annotations to save.")
        return

    if self.image_path:
        image_name = os.path.basename(self.image_path)
        json_filename = self.json_path if hasattr(self, 'json_path') else os.path.join(self.image_folder,
                                                                                       "annotations.json")
        if os.path.exists(json_filename):
            with open(json_filename, 'r') as f:
                data = json.load(f)
        else:
            data = {
                'images': [],
                'annotations': [],
                'categories': [
                    {
                        'id': 1,
                        'name': 'forest road',
                        'supercategory': 'none'
                    }
                ]
            }

        image_id = len(data['images']) + 1
        data['images'].append({
            'file_name': image_name,
            'height': self.image.shape[0],
            'width': self.image.shape[1],
            'id': image_id
        })
        for idx, annotation in enumerate(self.annotations):
            data['annotations'].append({
                'image_id': image_id,
                'segmentation': annotation['segmentation'],
                'category_id': annotation['category_id'],
                'id': len(data['annotations']) + 1
            })

        with open(json_filename, 'w') as f:
            json.dump(data, f, indent=4)
        messagebox.showinfo("Info", f"Annotations saved to {json_filename}")


def next_image(self, event=None):
    if self.image_list and self.current_image_index < len(self.image_list) - 1:
        self.current_image_index += 1
        self.load_image()


def prev_image(self):
    if self.image_list and self.current_image_index > 0:
        self.current_image_index -= 1
        self.load_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabelingApp(root)
    root.mainloop()
