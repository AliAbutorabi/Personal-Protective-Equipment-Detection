from ultralytics import YOLO
import cv2
import os

model = YOLO("ppe.pt")

results = model("./test", conf=0.1)

os.makedirs("output", exist_ok=True)

for i, result in enumerate(results):
    img = result.plot(font_size=0.5, line_width=2)

    cv2.imwrite(f"output/image_{i + 1}.jpg", img)
