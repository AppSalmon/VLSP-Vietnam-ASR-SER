#!/usr/bin/env python
# coding: utf-8
# nohup /opt/miniforge3/bin/python main.py > train.log 2>&1 &


# In[2]:


# !pip install --upgrade --quiet pip
# !pip install --upgrade --quiet datasets[audio] transformers accelerate evaluate jiwer tensorboard gradio


# In[3]:


# !pip install transformers==4.48.0


# In[7]:


from huggingface_hub import login
login(token="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")


# In[ ]:


# [Chỉnh sửa: Thêm biến điều khiển tập con và logic chọn số mẫu]
# Biến điều khiển tập con
use_subset = False  # Đặt False để dùng toàn bộ dataset
num_train_samples = 350  # Số mẫu train
num_test_samples = 200  # Số mẫu test


# In[4]:


# Tạo dataset từ file audio và transcript của VLSP2023
import os
from datasets import Dataset, DatasetDict, concatenate_datasets
import pandas as pd

def load_vlsp_dataset(audio_path, transcripts_path, use_subset=False, num_train_samples=1500, num_test_samples=200):
    # Đọc file transcript
    transcripts = []
    with open(transcripts_path, 'r', encoding='utf-8') as f:
        for line in f:
            audio_id, text = line.strip().split('\t')
            transcripts.append({
                'audio': os.path.join(audio_path, f"{audio_id}.wav"),
                'sentence': text
            })

    # Tạo dataset từ danh sách
    dataset = Dataset.from_list(transcripts)

    # Chia dataset thành train và test (80-20 nếu không dùng subset)
    train_test = dataset.train_test_split(test_size=0.2, seed=42)

    # Nếu dùng tập con, giới hạn số mẫu
    if use_subset:
        train_dataset = train_test["train"].select(range(min(num_train_samples, len(train_test["train"]))))
        test_dataset = train_test["test"].select(range(min(num_test_samples, len(train_test["test"]))))
        train_test = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })

    # Tạo DatasetDict
    vlsp_dataset = DatasetDict({
        "train": train_test["train"],
        "test": train_test["test"]
    })

    return vlsp_dataset


# In[5]:


# Tải dataset VLSP2023
audio_path = '/root/Dataset/vlsp_dataset2/vlsp2022-asr-task-1/wav'
transcripts_path = '/root/Dataset/vlsp_dataset2/vlsp2022-asr-task-1/transcript.txt'
vlsp_dataset = load_vlsp_dataset(audio_path, transcripts_path, use_subset=use_subset, num_train_samples=num_train_samples, num_test_samples=num_test_samples)


# In[6]:


print(vlsp_dataset)


# In[7]:


from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

# [Chỉnh sửa: Thay đổi ngôn ngữ tokenizer sang tiếng Việt]
from transformers import WhisperTokenizer

# Khởi tạo tokenizer với ngôn ngữ tiếng Việt
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Vietnamese", task="transcribe")


# [Chỉnh sửa: Thay đổi ngôn ngữ processor sang tiếng Việt]
from transformers import WhisperProcessor

# Khởi tạo processor với ngôn ngữ tiếng Việt
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Vietnamese", task="transcribe")



# In mẫu dữ liệu đầu tiên trong tập train
print(vlsp_dataset["train"][0])


from datasets import Audio

# [Chỉnh sửa: Đảm bảo sampling rate là 16kHz cho audio]
# Chuyển cột audio sang định dạng Audio với sampling rate 16kHz
# vlsp_dataset = vlsp_dataset.cast_column("audio", Audio(sampling_rate=16000))
# In lại mẫu dữ liệu đầu tiên sau khi xử lý audio
# print(vlsp_dataset["train"][0])



# !pip install torchaudio

#!/usr/bin/env python
# coding: utf-8

# Cài đặt các thư viện cần thiết
import os
import torch
from datasets import DatasetDict
import torchaudio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

# Giả định các biến global đã được định nghĩa từ code trước
# (feature_extractor, tokenizer, processor, vlsp_dataset)

# # Hàm chuẩn bị dataset cho training (tối ưu hóa với GPU và batch)
# def prepare_dataset(batch):
#     # Chuyển đổi audio sang tensor và đảm bảo sampling rate 16kHz
#     audio_arrays = []
#     for audio in batch["audio"]:
#         # Giải mã audio từ AudioDecoder và lấy mảng dữ liệu
#         waveform = audio["array"]  # Lấy mảng audio từ datasets.Audio
#         sample_rate = audio["sampling_rate"]

#         # Resample sang 16kHz nếu cần (giữ tensor PyTorch)
#         if sample_rate != 16000:
#             resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(waveform.device)
#             waveform = resampler(waveform)

#         # Chuyển sang numpy mà không cần .cpu() nếu đã là numpy
#         if isinstance(waveform, torch.Tensor):
#             waveform = waveform.squeeze().cpu().numpy()
#         else:
#             waveform = waveform.squeeze()  # Giả định đã là numpy

#         audio_arrays.append(waveform)

#     # Tính toán đặc trưng log-Mel từ mảng audio
#     batch["input_features"] = feature_extractor(audio_arrays, sampling_rate=16000).input_features

#     # Mã hóa văn bản thành label ids
#     batch["labels"] = tokenizer(batch["sentence"], padding=True, return_tensors="pt").input_ids

#     return batch

# # Áp dụng hàm prepare_dataset cho toàn bộ dataset với tối ưu hóa
# # Sử dụng batch_size để xử lý theo lô và tăng num_proc
# batch_size = 32  # Điều chỉnh batch_size tùy theo tài nguyên
# num_proc = min(32, os.cpu_count())  # Sử dụng tối đa 8 tiến trình hoặc số lõi CPU nếu ít hơn
# # num_proc = os.cpu_count()
# vlsp_dataset = vlsp_dataset.map(
#     prepare_dataset,
#     batched=True,  # Bật chế độ batch processing
#     batch_size=batch_size,
#     remove_columns=vlsp_dataset.column_names["train"],
#     num_proc=num_proc,
# )

# # In thông tin dataset sau khi xử lý
# print(vlsp_dataset)
# # Lưu dataset đã xử lý
# vlsp_dataset.save_to_disk("/root/Dataset/vlsp_processed")

# # Tải lại dataset khi cần
# from datasets import load_from_disk
# vlsp_dataset = load_from_disk("/root/Dataset/vlsp_processed")

# print(vlsp_dataset)

"""Vip Pro"""

# [Chỉnh sửa: Thêm import cần thiết và tối ưu hóa hàm prepare_dataset]
import os
import torch
import torchaudio
from datasets import Dataset, Audio

# Hàm kiểm tra và làm sạch dataset
# [Chỉnh sửa: Cập nhật validate_dataset để lọc audio theo thời lượng]
# [Chỉnh sửa: Cập nhật validate_dataset để xử lý AudioDecoder và lọc thời lượng]
def validate_dataset(dataset):
    valid_indices = []
    for i, sample in enumerate(dataset):
        audio_path = sample["audio"]  # [Chỉnh sửa: Sử dụng cột audio_path]
        sentence = sample["sentence"]
        try:
            # Kiểm tra file âm thanh tồn tại và đọc được
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                continue
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.numel() == 0:
                print(f"Empty audio file: {audio_path}")
                continue
            # Kiểm tra thời lượng âm thanh
            duration = waveform.shape[-1] / sample_rate
            if duration < 1.0 or duration > 30.0:
                print(f"Audio duration out of range ({duration:.2f}s): {audio_path}")
                continue
            if not isinstance(sentence, str) or len(sentence.strip()) == 0:
                print(f"Invalid sentence at index {i}: {sentence}")
                continue
            valid_indices.append(i)
        except Exception as e:
            print(f"Error validating sample {i}: {e}")
            continue
    print(f"Valid samples: {len(valid_indices)}/{len(dataset)}")
    return dataset.select(valid_indices)

# [Chỉnh sửa: Tối ưu hóa hàm prepare_dataset, loại bỏ resampling và thêm xử lý lỗi]
def prepare_dataset(batch):
    try:
        # Lấy mảng audio (đã đảm bảo sampling rate 16kHz từ trước)
        audio_arrays = []
        for audio in batch["audio"]:
            waveform = audio["array"]
            if waveform is None or len(waveform) == 0:
                raise ValueError("Invalid audio data")
            audio_arrays.append(waveform)

        # Tính toán đặc trưng log-Mel từ mảng audio
        batch["input_features"] = feature_extractor(audio_arrays, sampling_rate=16000).input_features

        # Mã hóa văn bản thành label ids
        batch["labels"] = tokenizer(batch["sentence"], padding=True, return_tensors="pt").input_ids
        return batch
    except Exception as e:
        print(f"Error processing batch: {e}")
        return None  # Bỏ qua batch lỗi

# [Chỉnh sửa: Thêm hàm xử lý dataset theo chunk]
def process_dataset_in_chunks(dataset, chunk_size=5000):
    processed_datasets = []
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset.select(range(i, min(i + chunk_size, len(dataset))))
        print(f"Processing chunk {i//chunk_size + 1}/{len(dataset)//chunk_size + 1}")
        processed_chunk = chunk.map(
            prepare_dataset,
            batched=True,
            batch_size=64,  # [Chỉnh sửa: Tăng batch_size để giảm số lượng batch]
            remove_columns=chunk.column_names,
            num_proc=4,  # [Chỉnh sửa: Giảm num_proc để tránh quá tải]
            drop_last_batch=True,  # Bỏ batch lỗi
        )
        processed_datasets.append(processed_chunk)
    return concatenate_datasets(processed_datasets)



# Kiểm tra và làm sạch dataset
print("Validating train dataset...")
vlsp_dataset["train"] = validate_dataset(vlsp_dataset["train"])
print("Validating test dataset...")
vlsp_dataset["test"] = validate_dataset(vlsp_dataset["test"])

# [Chỉnh sửa: Áp dụng kiểm tra dữ liệu và xử lý dataset]
# Đảm bảo sampling rate 16kHz (đã có trong code gốc)
vlsp_dataset = vlsp_dataset.cast_column("audio", Audio(sampling_rate=16000))

# Xử lý dataset theo chunk
print("Processing train dataset...")
vlsp_dataset["train"] = process_dataset_in_chunks(vlsp_dataset["train"], chunk_size=5000)
print("Processing test dataset...")
vlsp_dataset["test"] = process_dataset_in_chunks(vlsp_dataset["test"], chunk_size=5000)

# [Chỉnh sửa: Lưu dataset đã xử lý]
# vlsp_dataset.save_to_disk("/root/Dataset/vlsp_processed")
# print("Dataset processed and saved to /root/Dataset/vlsp_processed")
print(vlsp_dataset)
# In[ ]:

# Tải lại dataset khi cần
# from datasets import load_from_disk
# vlsp_dataset = load_from_disk("/root/Dataset/vlsp_processed")
print(vlsp_dataset)

"""End Vip Pro"""


# # Hàm chuẩn bị dataset cho training
# def prepare_dataset(batch):
#     # Tải và chuyển đổi audio từ 48kHz (nếu có) sang 16kHz
#     audio = batch["audio"]

#     # Tính toán đặc trưng log-Mel từ mảng audio
#     batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

#     # Mã hóa văn bản thành label ids
#     batch["labels"] = tokenizer(batch["sentence"]).input_ids
#     return batch

# # Áp dụng hàm prepare_dataset cho toàn bộ dataset
# vlsp_dataset = vlsp_dataset.map(prepare_dataset, remove_columns=vlsp_dataset.column_names["train"], num_proc=32)


# In[ ]:


from transformers import WhisperForConditionalGeneration

# Tải model Whisper small
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")


# In[ ]:


# [Chỉnh sửa: Cấu hình model cho tiếng Việt]
# Thiết lập ngôn ngữ và task cho model
model.generation_config.language = "vietnamese"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None


# In[18]:


import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Định nghĩa data collator để xử lý batch
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Tách input features và labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Lấy các chuỗi label đã được tokenize
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Thay thế padding bằng -100 để bỏ qua trong tính loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Loại bỏ token BOS nếu có
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# In[19]:


# Khởi tạo data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


# In[20]:


import evaluate

# Tải metric WER (Word Error Rate)
metric = evaluate.load("wer")

# Hàm tính toán metrics
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Thay -100 bằng pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Giải mã predictions và labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Tính WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# In[ ]:


from transformers import Seq2SeqTrainingArguments, EarlyStoppingCallback

# Cấu hình tham số huấn luyện
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-vi-vlsp",  # [Chỉnh sửa: Thay đổi output directory cho tiếng Việt]
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)



# In[22]:


from transformers import Seq2SeqTrainer

# Khởi tạo trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=vlsp_dataset["train"],
    eval_dataset=vlsp_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)


# In[23]:


# Lưu processor
processor.save_pretrained(training_args.output_dir)


# In[24]:


# Huấn luyện model
trainer.train()


# 1500_vlsp_1: 27.645639 \
# 1500_vlsp_2: 25.680800 \
# 1500_vlsp_3_flash: 26.611513

# In[32]:


# # [Chỉnh sửa: Cập nhật metadata cho model tiếng Việt]
# kwargs = {
#     "dataset_tags": "vlsp2023",
#     "dataset": "VLSP 2023 ASR",  # Tên dataset
#     "dataset_args": "split: test",
#     "language": "vi",
#     "model_name": "Whisper Small Vi - Salmon173",  # [Chỉnh sửa: Thay đổi tên model]
#     "finetuned_from": "openai/whisper-small",
#     "tasks": "automatic-speech-recognition",
# }


# In[ ]:


# trainer.save_model() 
# trainer.push_to_hub() 
# tokenizer.push_to_hub("username/model-id") 


# In[33]:


# Đẩy model lên Hugging Face Hub
trainer.push_to_hub()


# In[46]:


# # [Chỉnh sửa: Cập nhật pipeline và Gradio cho tiếng Việt]
# from transformers import pipeline
# import gradio as gr

# # Tạo pipeline cho model đã fine-tune
# pipe = pipeline(model="SalmonAI123/whisper-small-vi")  # [Chỉnh sửa: Thay đổi thành tên model của bạn]

# # Hàm chuyển đổi audio thành văn bản
# def transcribe(audio):
#     text = pipe(audio)["text"]
#     return text

# # Tạo giao diện Gradio
# iface = gr.Interface(
#     fn=transcribe,
#     # inputs=gr.Audio(source="microphone", type="filepath"),
#     inputs=gr.Audio(sources=["microphone"]),
#     outputs="text",
#     title="Whisper Small Vietnamese",
#     description="Demo nhận diện giọng nói tiếng Việt thời gian thực sử dụng model Whisper small đã được fine-tune.",
# )

# # Chạy giao diện
# iface.launch()


# In[25]:


#!/usr/bin/env python
# coding: utf-8

# Cài đặt các thư viện cần thiết
import os
import glob
import zipfile
from transformers import pipeline
import torch

# Đường dẫn đến thư mục chứa file âm thanh public test
test_audio_dir = "/root/Dataset/public_test/wav"

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
    transcription = pipe(audio_path)["text"]

    # Chuẩn hóa văn bản (chữ thường)
    transcription = normalize_text(transcription)

    # Thêm kết quả vào danh sách với emotion_label mặc định là Positive
    results.append(f"{utterance_name}\tPositive\t{transcription}")

# Lưu kết quả vào file results.tsv
output_tsv = "results.tsv"
with open(output_tsv, "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")

# Nén file results.tsv thành file zip
output_zip = "submission.zip"
with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(output_tsv)

print(f"Đã tạo file {output_tsv} và nén thành {output_zip}")
print(f"Sẵn sàng upload file {output_zip} lên AI Hub")


# In[ ]:




