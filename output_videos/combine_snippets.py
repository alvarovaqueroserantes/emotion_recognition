import cv2
import os
import random

# === Lista de rutas de videos ===
video_paths = [
    "output_video_2.mp4",
    "output_video_3.mp4",
    "output_video_5.mp4",
    "output_video_6.mp4",
    "output_video_7.mp4",
]

clip_duration_sec = 5  # Duración por fragmento
output_video = "final_linkedin_video.mp4"
output_fps = 25

# === Obtener resolución del primer video válido ===
for ref_path in video_paths:
    if os.path.exists(ref_path):
        cap_test = cv2.VideoCapture(ref_path)
        if cap_test.isOpened():
            width = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_size = (width, height)
            cap_test.release()
            break
else:
    raise RuntimeError("No hay videos válidos.")

# === Usar códec H.264 (si soportado) o MJPG como alternativa
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # o 'MJPG' si avc1 no funciona
out = cv2.VideoWriter(output_video, fourcc, output_fps, output_size)

for path in video_paths:
    if not os.path.exists(path):
        print(f"❌ No encontrado: {path}")
        continue

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"⚠️ No se pudo abrir: {path}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_capture = int(min(clip_duration_sec * fps, total_frames))

    max_start = total_frames - frames_to_capture
    if max_start <= 0:
        print(f"⚠️ {path} muy corto para extraer {clip_duration_sec} segundos")
        cap.release()
        continue

    start_frame = random.randint(0, max_start)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_written = 0
    while frames_written < frames_to_capture:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, output_size)  # opcional si videos no coinciden
        out.write(frame_resized)
        frames_written += 1

    cap.release()
    print(f"✅ Clip añadido desde {path}")

out.release()
print(f"\n🎬 Video final exportado como {output_video}")
