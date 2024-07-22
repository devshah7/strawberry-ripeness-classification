from ultralytics import YOLO

data_yaml = './Dataset_4/data.yaml'

model = YOLO()

model.train( data = data_yaml, epochs = 100)

# The new model get stored in ./run/detect/train/weights folder as best.pt file. 
# You can rename it and/or move it to the ./models folder for better access.