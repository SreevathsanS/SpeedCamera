import onnxruntime as ort
import cv2
import numpy as np

# Paths
onnx_model_path = 'best.onnx'  # Your model path
image_path = 'plate_boxed.jpg'        # Image to run detection on
output_path = 'output.jpg'     # Output result

# Load model
session = ort.InferenceSession(onnx_model_path)

# Read and preprocess image
original_image = cv2.imread(image_path)
input_image = cv2.resize(original_image, (640, 640))
input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_image_np = input_image_rgb.astype(np.float32) / 255.0
input_image_np = np.transpose(input_image_np, (2, 0, 1))  # CHW
input_tensor = input_image_np[np.newaxis, :]  # Add batch dimension

# Run inference
output = session.run(None, {session.get_inputs()[0].name: input_tensor})[0]  # shape: (1, 5, 8400)
output = np.squeeze(output, axis=0)  # shape: (5, 8400)

# Postprocess
conf_threshold = 0.25
iou_threshold = 0.45

boxes = []
confidences = []

for i in range(output.shape[1]):
    confidence = output[4, i]
    if confidence > conf_threshold:
        x_center, y_center, w, h = output[0:4, i]

        # Convert to pixel coordinates
        x = int((x_center - w / 2) * original_image.shape[1] / 640)
        y = int((y_center - h / 2) * original_image.shape[0] / 640)
        width = int(w * original_image.shape[1] / 640)
        height = int(h * original_image.shape[0] / 640)

        boxes.append([x, y, width, height])
        confidences.append(float(confidence))

# Apply NMS
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)

# Draw boxes
for i in indices:
    i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
    x, y, w, h = boxes[i]
    label = f"Plate {confidences[i]:.2f}"
    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(original_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

# Save the result
cv2.imwrite(output_path, original_image)
print(f"[✅] Output saved to {output_path}")
