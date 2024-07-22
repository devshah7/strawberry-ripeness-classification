from ultralytics import YOLO

# Choose the dataset you want to test and provide its source path or .yaml path.
data_yaml = './Dataset_4/data.yaml'

# Choose the model you want to test and provide its source path.
model = YOLO('./models/model_1.pt')

model.val(data = data_yaml)

# The validation results are stored in ./run/detect/val folder.