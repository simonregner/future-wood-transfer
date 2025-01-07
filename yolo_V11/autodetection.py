from ultralytics.data.annotator import auto_annotate

auto_annotate(data='../../data/GOOSE/images', det_model="runs/segment/train_with_asphalt/weights/best.pt", sam_model='sam2_b.pt', output_dir='../../data/GOOSE/annotation')