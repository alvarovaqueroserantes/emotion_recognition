import cv2
import torch
import yaml
import os
from torchvision import transforms
from models.emotion_cnn import get_model
from PIL import Image
import cvlib as cv

# === Configuración del modelo ===
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")

# === Cargar modelo entrenado ===
model = get_model(config["model_name"], config["num_classes"])
checkpoint = torch.load(config["checkpoint_path"], map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

# === Transformaciones ===
transform = transforms.Compose([
    transforms.Resize((config["input_size"], config["input_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# === Lista de videos a procesar ===
video_paths = [
r"C:\Users\alvar\Documents\emotion_recognition\final_linkedin_video.avi"
]

# === Procesar cada video ===
for idx, video_path in enumerate(video_paths):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir: {video_path}")
        continue

    # Info para escritura de video de salida
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_path = f"output_video_{idx+1}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (960, 540))

    print(f"🔁 Procesando: {os.path.basename(video_path)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces, confidences = cv.detect_face(frame)

        for (box, conf) in zip(faces, confidences):
            if conf < 0.60:
                continue  # solo caras con ≥90%

            x1, y1, x2, y2 = box
            face_roi = frame[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                label = EMOTION_CLASSES[pred.item()]

            # Dibujar caja y etiqueta
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.0%})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Redimensionar y guardar
        resized_frame = cv2.resize(frame, (960, 540))
        out_writer.write(resized_frame)

        # Mostrar preview
        cv2.imshow(f"Processing {idx+1}", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    print(f"✅ Guardado: {out_path}")

print("✅ Todos los videos procesados.")
