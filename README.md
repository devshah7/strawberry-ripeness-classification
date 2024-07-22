# Strawberry Ripeness Project

## Steps -
1. Clone the project from the repo and open it in VS Code or any other code editor.

2. Open new terminal in the VS code or terminal under the project folder (we'll be performing train/val/predict operations in the terminal).

3. Install Ultralytics.
```
pip install ultralytics
```
3. For training a new model, you can make changes to train_model.py file using hyperparameters and call it in the terminal.
```
python train_model.py
```
4. For validating the model, call the val_model.py file.
```
python val_model.py
```
5. For testing the model for predictions, call test_model.py file. Change the source value based on the data you want to test on.
```
python test_model.py
```
6. For testing the model for predictions on live camera, call test_model_live.py file. You can modify the label colors, font, size, etc.
```
python test_model_live.py
```