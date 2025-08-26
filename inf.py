# Cài đặt các thư viện cần thiết
import os
import glob
import zipfile
from transformers import pipeline
import torch

# Đường dẫn đến thư mục chứa file âm thanh public test
test_audio_dir = "/root/wav-private/wav"

# Đường dẫn đến model đã fine-tune
model_path = "./whisper-small-vi-vlsp"  # Thay bằng đường dẫn đến model của bạn nếu khác

# Tạo pipeline cho nhận diện giọng nói
pipe = pipeline("automatic-speech-recognition", model=model_path, device=0 if torch.cuda.is_available() else -1)

# Hàm chuyển đổi văn bản thành chữ thường và loại bỏ ký tự không cần thiết
def normalize_text(text):
    return text.lower().strip()

# Tạo danh sách kết quả cho file results.tsv
results = []
cnt = 0
# Duyệt qua tất cả file .wav trong thư mục public test
for audio_path in glob.glob(os.path.join(test_audio_dir, "*.wav")):
    # Lấy tên file (utterance_name)
    cnt += 1
    if cnt % 100 == 0:
        print("Cnt:", cnt)
    utterance_name = os.path.basename(audio_path)

    # Dự đoán văn bản từ file âm thanh
    # transcription = pipe(audio_path)["text"]
    transcription = pipe(
    audio_path,
    generate_kwargs={
        "num_beams": 5,
        "length_penalty": 0.9,
        "do_sample": False,
        "no_repeat_ngram_size": 5,
        "repetition_penalty": 1.5
        }
    )["text"]


    # Chuẩn hóa văn bản (chữ thường)
    transcription = normalize_text(transcription)

    # Thêm kết quả vào danh sách với emotion_label mặc định là Positive
    results.append(f"{utterance_name}\tPositive\t{transcription}")

# Lưu kết quả vào file results.tsv
output_tsv = "results_pro.tsv"
with open(output_tsv, "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")

# Nén file results.tsv thành file zip
output_zip = "submission.zip"
with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(output_tsv)

print(f"Đã tạo file {output_tsv} và nén thành {output_zip}")
print(f"Sẵn sàng upload file {output_zip} lên AI Hub")
