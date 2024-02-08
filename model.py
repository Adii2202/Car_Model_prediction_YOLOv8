from ultralytics import YOLO
#Load a pretrained model
model = YOLO('best.pt')
#Run the inference
result  = model(source=0, show=True, conf=0.6, save=True)

