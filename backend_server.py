from fastapi import FastAPI, Request
import base64
import os
import subprocess
import torch
import torchaudio
from pydub import AudioSegment
from transformers import AutoModel, AutoProcessor
from pathlib import Path

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MOSS_MODEL_ID = "OpenMOSS-Team/MOSS-TTS-v1.5"

# Предзагрузка моделей в память GPU при старте сервера
try:
    processor = AutoProcessor.from_pretrained(MOSS_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MOSS_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16).to(DEVICE)
    model.eval()
except Exception as e:
    print(f"Ошибка загрузки MOSS-TTS: {e}")

def execute_vocal_isolation(input_path, output_dir):
    cmd = ["demucs", "-n", "htdemucs_ft", "--two-stems=vocals", "-o", output_dir, input_path]
    subprocess.run(cmd, check=True)
    base_name = Path(input_path).stem
    return os.path.join(output_dir, "htdemucs_ft", base_name, "vocals.wav"), os.path.join(output_dir, "htdemucs_ft", base_name, "no_vocals.wav")

@app.post("/dub")
async def process_dubbing(request: Request):
    data = await request.json()
    audio_b64 = data.get("audio_base64")
    segments = data.get("segments")
    
    input_path = "/workspace/source_audio.aac"
    with open(input_path, "wb") as f:
        f.write(base64.b64decode(audio_b64))
        
    try:
        out_dir = "/workspace/demucs_out"
        vocals_path, bg_path = execute_vocal_isolation(input_path, out_dir)
        
        orig_vocals = AudioSegment.from_file(vocals_path)
        bg_audio = AudioSegment.from_file(bg_path)
        dub_canvas = AudioSegment.silent(duration=len(bg_audio))
        
        for seg in segments:
            start_ms, end_ms = int(seg["start"] * 1000), int(seg["end"] * 1000)
            duration_s = seg["end"] - seg["start"]
            
            ref_path = f"/workspace/ref_{start_ms}.wav"
            orig_vocals[start_ms:end_ms].export(ref_path, format="wav")
            ref_tensor, _ = torchaudio.load(ref_path)
            
            target_tokens = int(duration_s * 12.5)
            msgs = [processor.build_user_message(text=seg["text"], reference=[ref_tensor], tokens=target_tokens)]
            inputs = processor([msgs], return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=target_tokens + 50, audio_temperature=1.7)
            
            audio_codes = processor.decode(outputs)[0].audio_codes_list[0]
            waveform = processor.decode_audio(audio_codes)
            
            synth_path = f"/workspace/synth_{start_ms}.wav"
            
            # Обеспечиваем правильную размерность тензора перед сохранением. 
            # torchaudio.save ожидает 2D тензор размерности [channels, frames].
            waveform_tensor = torch.tensor(waveform).cpu()
            if waveform_tensor.dim() == 1:
                waveform_tensor = waveform_tensor.unsqueeze(0)
                
            torchaudio.save(synth_path, waveform_tensor, 48000)
            
            synth_segment = AudioSegment.from_file(synth_path)
            dub_canvas = dub_canvas.overlay(synth_segment, position=start_ms)
            
            # Очистка кэша CUDA после каждой итерации для предотвращения утечек памяти (OOM) при длинных аудиофайлах
            torch.cuda.empty_cache()
            
        final_comp = bg_audio.overlay(dub_canvas)
        out_file = "/workspace/final.aac"
        final_comp.export(out_file, format="adts")
        
        with open(out_file, "rb") as f:
            encoded_result = base64.b64encode(f.read()).decode('utf-8')
            
        return {"status": "success", "audio_base64": encoded_result}
    except Exception as e:
        return {"status": "error", "error": str(e)}
