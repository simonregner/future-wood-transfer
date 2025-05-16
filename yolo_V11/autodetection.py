from ultralytics.data.annotator import auto_annotate

auto_annotate(data='../../SVO/MM_ForestRoads_01/images', det_model="runs/segment/MM_Dataset/weights/best.pt", sam_model='sam2_b.pt', output_dir='../../SVO/MM_ForestRoads_01/annotation', classes=[2, 3, 4, 5], conf=0.7, iou=0.0)