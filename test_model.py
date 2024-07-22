from ultralytics import YOLO

# Choose the images file, image folder, or video file and provide its path. 

data_path = './Dataset_4/test/images'

# Choose the model you want to test and provide its source path.

model = YOLO('./models/model_1.pt')

# Input source=0 for live camera feed. You can also test using test_model_live.py file.

model.predict(source=data_path, save=True)

# The prediction results are stored in ./run/detect/predict folder.