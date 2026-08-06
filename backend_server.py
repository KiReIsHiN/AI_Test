import os
import base64
import threading
import subprocess
import importlib.util
import torch
import torchaudio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from pydub import AudioSegment
from transformers import AutoModel, AutoProcessor
from pathlib import Path

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ИСПРАВЛЕНИЕ: "OpenMOSS-Team/MOSS-TTS-v1.5" не существует в официальном
# репозитории OpenMOSS/MOSS-TTS — там опубликована модель под именем
# "OpenMOSS-Team/MOSS-TTS" (MossTTSDelay, 8B, флагманская для продакшена).
# Источник: https://github.com/OpenMOSS/MOSS-TTS (Released Models table) и
# https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_model_card.md
MOSS_MODEL_ID = "OpenMOSS-Team/MOSS-TTS"
TOKENS_PER_SECOND = 12.5  # задокументировано в model card: "1s ≈ 12.5 tokens"
INSTALL_LOG_PATH = "/workspace/install.log"

# Секрет для авторизации /dub. Задаётся app.py как env-переменная при
# старте пода. Если не задан — эндпоинт открыт всем, кто узнает URL пода,
# так что явно предупреждаем в логах, а не молчим об этом.
DUB_AUTH_TOKEN = os.environ.get("DUB_AUTH_TOKEN")
if not DUB_AUTH_TOKEN:
    print("[WARNING] DUB_AUTH_TOKEN не задан — эндпоинт /dub открыт БЕЗ авторизации.")


def _resolve_attn_implementation(device: str, dtype) -> str:
    # Точь-в-точь логика выбора backend'а из официального README MOSS-TTS.
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if device == "cuda":
        return "sdpa"
    return "eager"


# ИСПРАВЛЕНИЕ: официальный README явно отключает "сломанный" cuDNN SDPA
# backend перед загрузкой модели, иначе возможны сбои/нестабильность на
# некоторых связках driver/cuDNN. В прошлой версии этого не было.
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
ATTN_IMPL = _resolve_attn_implementation(DEVICE, DTYPE)

processor = None
model = None
MODEL_LOAD_ERROR = None


def _load_model_in_background():
    """
    ИСПРАВЛЕНИЕ: раньше загрузка модели шла прямо в теле модуля, то есть
    ДО того, как uvicorn успевал забиндить порт и начать отвечать на
    запросы. На это уходят минуты (скачивание + загрузка 8B весов в
    VRAM), и всё это время порт 5000 ничем не обслуживался — снаружи это
    выглядело как зависшая на 404 консоль логов.

    Теперь загрузка идёт в фоновом потоке: uvicorn поднимается почти
    сразу после pip install, /install.log и /health отвечают сразу же,
    а /dub возвращает 503 "модель ещё грузится", пока model is None.
    """
    global processor, model, MODEL_LOAD_ERROR
    try:
        print("[INFO] Загрузка MOSS-TTS начата в фоновом потоке...")
        proc = AutoProcessor.from_pretrained(MOSS_MODEL_ID, trust_remote_code=True)
        proc.audio_tokenizer = proc.audio_tokenizer.to(DEVICE)
        mdl = AutoModel.from_pretrained(
            MOSS_MODEL_ID,
            trust_remote_code=True,
            attn_implementation=ATTN_IMPL,
            torch_dtype=DTYPE,
        ).to(DEVICE)
        mdl.eval()
        processor = proc
        model = mdl
        print("[INFO] MOSS-TTS загружена, /dub готов принимать запросы.")
    except Exception as e:
        MODEL_LOAD_ERROR = str(e)
        print(f"[ERROR] Не удалось загрузить MOSS-TTS: {e}")


threading.Thread(target=_load_model_in_background, daemon=True).start()


@app.get("/health")
async def health():
    return {
        "model_ready": model is not None and processor is not None,
        "error": MODEL_LOAD_ERROR,
    }


@app.get("/install.log", response_class=PlainTextResponse)
async def get_install_log():
    # Тот же файл, в который шелл-скрипт пода пишет вывод apt/pip И вывод
    # самого uvicorn (весь запуск обёрнут в `{ ... } >> install.log 2>&1`
    # в app.py) — то есть этот эндпоинт продолжает без разрыва то, что до
    # него отдавал временный python -m http.server на этапе pip install.
    if not os.path.exists(INSTALL_LOG_PATH):
        return "(install.log ещё не создан)"
    with open(INSTALL_LOG_PATH, "r", errors="ignore") as f:
        return f.read()


def execute_vocal_isolation(input_path, output_dir):
    cmd = ["demucs", "-n", "htdemucs_ft", "--two-stems=vocals", "-o", output_dir, input_path]
    subprocess.run(cmd, check=True)
    base_name = Path(input_path).stem
    return (
        os.path.join(output_dir, "htdemucs_ft", base_name, "vocals.wav"),
        os.path.join(output_dir, "htdemucs_ft", base_name, "no_vocals.wav"),
    )


@app.post("/dub")
async def process_dubbing(request: Request):
    if DUB_AUTH_TOKEN:
        auth_header = request.headers.get("authorization", "")
        if auth_header != f"Bearer {DUB_AUTH_TOKEN}":
            raise HTTPException(status_code=401, detail="Invalid or missing Authorization header")

    if model is None or processor is None:
        detail = MODEL_LOAD_ERROR or "Модель MOSS-TTS ещё грузится в VRAM, подождите и повторите запрос."
        raise HTTPException(status_code=503, detail=detail)

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

            target_tokens = max(1, round(duration_s * TOKENS_PER_SECOND))

            # ИСПРАВЛЕНИЕ: по документации UserMessage.reference — это
            # List[str] (путь к файлу или URL), а не загруженный тензор.
            # ИСПРАВЛЕНИЕ: вызов processor(...) должен идти с mode="generation"
            # и без return_tensors="pt" — так задокументирован API кастомного
            # processor'а MOSS-TTS (README quickstart / model card).
            conversation = [
                processor.build_user_message(text=seg["text"], reference=[ref_path], tokens=target_tokens)
            ]
            batch = processor([conversation], mode="generation")
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            max_new_tokens = int(target_tokens * 1.5) + 64

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    audio_temperature=1.7,
                    audio_top_p=0.8,
                    audio_top_k=25,
                    audio_repetition_penalty=1.0,
                )

            # ИСПРАВЛЕНИЕ: официальный пример НЕ вызывает отдельный
            # processor.decode_audio(...) — такого метода в документации
            # нет. processor.decode(outputs)[i].audio_codes_list[0] уже
            # является готовым waveform-тензором, который можно сохранять
            # напрямую через torchaudio.save (см. README quickstart).
            message = processor.decode(outputs)[0]
            waveform_tensor = message.audio_codes_list[0]
            if waveform_tensor.dim() == 1:
                waveform_tensor = waveform_tensor.unsqueeze(0)
            waveform_tensor = waveform_tensor.cpu()

            synth_path = f"/workspace/synth_{start_ms}.wav"

            # ИСПРАВЛЕНИЕ: раньше частота дискретизации была захардкожена в
            # 48000. MOSS-Audio-Tokenizer (используется флагманской MOSS-TTS,
            # не Nano) в README описан как работающий с 24kHz-аудио — берём
            # реальное значение из конфига процессора, а не гадаем.
            sample_rate = processor.model_config.sampling_rate
            torchaudio.save(synth_path, waveform_tensor, sample_rate)

            synth_segment = AudioSegment.from_file(synth_path)
            dub_canvas = dub_canvas.overlay(synth_segment, position=start_ms)

            torch.cuda.empty_cache()

        final_comp = bg_audio.overlay(dub_canvas)
        out_file = "/workspace/final.aac"
        final_comp.export(out_file, format="adts")

        with open(out_file, "rb") as f:
            encoded_result = base64.b64encode(f.read()).decode('utf-8')

        return {"status": "success", "audio_base64": encoded_result}
    except Exception as e:
        return {"status": "error", "error": str(e)}
