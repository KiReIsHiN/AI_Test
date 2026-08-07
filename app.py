import io
import os
import time
import base64
import tempfile
import subprocess
import secrets as pysecrets

import runpod
import requests
import streamlit as st
from openai import OpenAI


def convert_to_wav_bytes(raw_bytes: bytes) -> bytes:
    """
    Перекодирует произвольный аудиофайл в WAV через системный ffmpeg
    (поставлен через packages.txt).

    ИСПРАВЛЕНИЕ: раньше здесь стоял pydub.AudioSegment, но pydub на
    Python 3.13+ падает при импорте, потому что тянет за собой модуль
    audioop, а тот убрали из стандартной библиотеки Python (PEP 594);
    fallback pydub на pyaudioop тоже не работает — пакета с таким именем
    в PyPI нет (см. github.com/jiaaro/pydub issues #863, #815, открыты).
    Streamlit Cloud сейчас на Python 3.14, поэтому уперлись сразу. Прямой
    вызов ffmpeg такой проблемы не имеет и, по сути, делает то же самое.
    """
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input")
        out_path = os.path.join(tmp, "output.wav")
        with open(in_path, "wb") as f:
            f.write(raw_bytes)

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, out_path],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg не смог перекодировать файл: {result.stderr.decode(errors='ignore')[-1000:]}")

        with open(out_path, "rb") as f:
            return f.read()

st.set_page_config(page_title="AiDubbing V4", layout="centered")
st.title("🎙️ AiDubbing V4: Управление и Дубляж")

API_PORT = 5000   # единственный порт: /install.log, /health, /dub — всё на нём

# --- Ключи и токены -----------------------------------------------------
with st.expander("🔑 Настройки API ключей", expanded=True):
    openai_key = st.text_input("OpenAI API Key", type="password")
    runpod_key = st.text_input("RunPod API Key", type="password")

    if "dub_auth_token" not in st.session_state:
        st.session_state.dub_auth_token = pysecrets.token_urlsafe(24)
    dub_auth_token = st.text_input(
        "Секрет для авторизации /dub (генерируется автоматически)",
        key="dub_auth_token",
        help="Этот же секрет передаётся в под как переменная окружения DUB_AUTH_TOKEN.",
    )

if not openai_key or not runpod_key:
    st.warning("Пожалуйста, введите оба ключа для начала работы.")
    st.stop()

runpod.api_key = runpod_key
llm_client = OpenAI(api_key=openai_key)

st.header("1. Управление GPU-сервером")

col1, col2 = st.columns(2)
with col1:
    selected_gpu = st.selectbox(
        "Тип GPU:",
        ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090", "NVIDIA RTX A6000", "NVIDIA RTX A5000", "NVIDIA RTX A4000"]
    )
with col2:
    selected_cloud = st.selectbox(
        "Тип облака:",
        ["COMMUNITY", "SECURE"],
        help="COMMUNITY - дешевле и доступно больше серверов. SECURE - корпоративные дата-центры, но часто бывают заняты."
    )

if st.button("🚀 Создать сервер, установить окружение и запустить"):
    with st.spinner("Запрос к RunPod..."):
        # ИСПРАВЛЕНИЕ (после двух попыток): второй порт (5001) для логов не
        # пробросился через прокси RunPod у вас на практике — не гадаю
        # дальше, почему именно, а убираю саму необходимость в нём.
        # Возвращаемся к ОДНОМУ порту 5000. Временный http.server отдаёт
        # install.log только пока идёт pip install (это и раньше работало
        # у вас нормально). Как только окружение готово, мы его убиваем и
        # стартует uvicorn — но теперь backend_server.py грузит модель в
        # фоновом потоке (см. CHANGES.md) и сам отвечает на /install.log
        # (читая тот же файл) и /health ПОЧТИ СРАЗУ после старта, а не
        # только после того как модель полностью загрузится в VRAM.
        # Поэтому "слепого окна" между двумя серверами больше не должно
        # быть, и второй порт не нужен.
        raw_script = f"""
        cd /workspace
        touch install.log

        python3 -m http.server {API_PORT} --directory /workspace &
        HTTP_PID=$!

        {{
            set -x
            echo "=== СТАРТ ИНИЦИАЛИЗАЦИИ ==="
            export DEBIAN_FRONTEND=noninteractive
            apt-get update
            apt-get install -y ffmpeg
            if [ ! -d /workspace/venv ]; then
                echo "=== СОЗДАНИЕ VENV И УСТАНОВКА ЗАВИСИМОСТЕЙ ==="
                python3 -m venv /workspace/venv
                /workspace/venv/bin/pip install fastapi uvicorn pydub demucs==4.1.0 "transformers>=5.0.0" accelerate orjson requests
                /workspace/venv/bin/pip install torch==2.9.1+cu128 torchaudio==2.9.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
                git clone https://github.com/OpenMOSS/MOSS-TTS.git
                cd MOSS-TTS
                /workspace/venv/bin/pip install -e .
                cd ..
            else
                echo "=== ОКРУЖЕНИЕ УЖЕ СУЩЕСТВУЕТ ==="
            fi
            echo "=== ЗАГРУЗКА БЕКЭНДА ==="
            wget -qO backend_server.py https://raw.githubusercontent.com/KiReIsHiN/AI_Test/main/backend_server.py
            echo "=== PIP-ЗАВИСИМОСТИ ГОТОВЫ. ПЕРЕДАЧА ПОРТА {API_PORT} ОТ ВРЕМЕННОГО СЕРВЕРА К FASTAPI ==="
        }} >> install.log 2>&1

        kill $HTTP_PID
        export PYTHONUNBUFFERED=1
        export DUB_AUTH_TOKEN='{dub_auth_token}'
        /workspace/venv/bin/python -m uvicorn backend_server:app --host 0.0.0.0 --port {API_PORT} >> install.log 2>&1
        """

        encoded_script = base64.b64encode(raw_script.encode('utf-8')).decode('utf-8')
        startup_script = f"bash -c 'echo {encoded_script} | base64 -d | bash'"

        try:
            pod = runpod.create_pod(
                name="AiDubbing-GPU-Backend",
                image_name="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
                gpu_type_id=selected_gpu,
                cloud_type=selected_cloud,
                volume_in_gb=50,
                container_disk_in_gb=20,
                ports=f"{API_PORT}/http",
                docker_args=startup_script
            )
            st.session_state['pod_id'] = pod['id']
            st.success(f"✅ Сервер создан! ID: {pod['id']}")
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка при создании сервера: {e}")

if "pod_id" not in st.session_state:
    st.session_state.pod_id = ""
pod_id = st.text_input("ID запущенного Pod'а (подставится автоматически):", key="pod_id")

if pod_id:
    backend_url = f"https://{pod_id}-{API_PORT}.proxy.runpod.net"
    dub_endpoint = f"{backend_url}/dub"

    st.header("2. Статус установки сервера")

    if "watch_logs" not in st.session_state:
        st.session_state.watch_logs = True
    watch = st.checkbox("Автообновление логов (каждые 3с)", key="watch_logs")

    @st.fragment(run_every="3s" if watch else None)
    def log_console():
        log_url = f"{backend_url}/install.log?t={int(time.time())}"
        try:
            res = requests.get(log_url, timeout=5)
            if res.status_code == 200:
                text = res.text or "(лог пока пуст)"
                st.code(text, language="bash", height=400)
            else:
                st.code(f"Сервер логов пока не отвечает (код {res.status_code}) — под ещё поднимается.",
                        language="bash", height=200)
                return
        except requests.exceptions.RequestException:
            st.code("Контейнер ещё скачивается/инициализируется на физической ноде...", language="bash", height=200)
            return

        # Готовность проверяем отдельно через /health, а не текстом в
        # логах — это то, что реально проверяет backend (model is not None),
        # а не догадка по фразам в stdout.
        try:
            health = requests.get(f"{backend_url}/health", timeout=5).json()
            if health.get("model_ready"):
                st.success("Модель загружена — сервер готов принимать /dub.")
            elif health.get("error"):
                st.error(f"Загрузка модели упала с ошибкой: {health['error']}")
            else:
                st.info("FastAPI поднялся, модель ещё грузится в VRAM — обычно несколько минут.")
        except requests.exceptions.RequestException:
            st.info("FastAPI ещё не поднялся (либо ещё идёт pip install, либо порт переключается).")

    log_console()

    st.header("3. Загрузка и Дубляж")
    audio_file = st.file_uploader("Загрузите исходный файл (.aac, .mp3, .m4a, .wav)")

    if audio_file is not None:
        if st.button("Инициировать дубляж"):
            with st.status("Оркестрация процесса...", expanded=True):
                st.write("1. Выполнение транскрипции (OpenAI Whisper)...")

                # ИСПРАВЛЕНИЕ: OpenAI transcriptions API официально принимает
                # только flac/mp3/mp4/mpeg/mpga/m4a/ogg/wav/webm — .aac туда
                # НЕ входит. Раньше файл просто переименовывался в .m4a без
                # перекодирования, из-за чего реальный AAC-поток не совпадал
                # с ожидаемым M4A-контейнером -> 400 BadRequestError. Теперь
                # файл всегда честно перекодируется в WAV через прямой вызов
                # ffmpeg (см. convert_to_wav_bytes выше — без pydub).
                raw_bytes = audio_file.getvalue()
                try:
                    wav_bytes = convert_to_wav_bytes(raw_bytes)
                except Exception as e:
                    st.error(f"Не удалось декодировать аудиофайл через ffmpeg: {e}")
                    st.stop()

                transcription_data = llm_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )

                st.write("2. Адаптивный перевод (с учетом тайминга)...")
                payload_segments = []

                for segment in transcription_data.segments:
                    start_time = segment.start
                    end_time = segment.end
                    dur = end_time - start_time
                    if dur < 0.6:
                        continue

                    max_chars = int(dur * 14)
                    prompt = f"Переведи на русский для дубляжа. ОГРАНИЧЕНИЕ: не более {max_chars} символов. Только текст."
                    resp = llm_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": f"{prompt}\nТекст: {segment.text}"}],
                        temperature=0.25
                    )
                    translated = resp.choices[0].message.content.strip()
                    payload_segments.append({"start": start_time, "end": end_time, "text": translated})
                    st.write(f"[{start_time:.1f}s - {end_time:.1f}s] {translated}")

                st.write("3. Отправка на GPU-сервер. Ожидание генерации...")
                encoded_audio = base64.b64encode(raw_bytes).decode('utf-8')

                try:
                    response = requests.post(
                        dub_endpoint,
                        json={"audio_base64": encoded_audio, "segments": payload_segments},
                        headers={"Authorization": f"Bearer {dub_auth_token}"},
                        timeout=600
                    )

                    if response.status_code == 200:
                        result_data = response.json()
                        if result_data.get("status") == "success":
                            final_audio_bytes = base64.b64decode(result_data["audio_base64"])
                            st.success("Дубляж завершен!")
                            st.audio(final_audio_bytes, format='audio/aac')
                            st.download_button("Скачать результат", final_audio_bytes, "dubbed.aac", "audio/aac")
                        else:
                            st.error(f"Ошибка GPU: {result_data.get('error')}")
                    elif response.status_code == 401:
                        st.error("401: секрет DUB_AUTH_TOKEN не совпадает с тем, что задан в поде (пересоздайте под).")
                    elif response.status_code == 503:
                        st.error(f"503: модель ещё не готова — {response.json().get('detail', '')}. Подождите и повторите.")
                    else:
                        st.error(f"Сервер недоступен (Код {response.status_code}). Убедитесь, что установка завершена.")
                except requests.exceptions.Timeout:
                    st.error("Превышено время ожидания ответа от сервера. Процесс генерации занимает много времени.")

    st.markdown("---")

    def _stop_pod():
        # ИСПРАВЛЕНИЕ: раньше session_state['pod_id'] менялся ПОСЛЕ того,
        # как в этом же прогоне уже был создан виджет text_input(key="pod_id")
        # — Streamlit это запрещает и падает с StreamlitAPIException.
        # Колбэк on_click выполняется ДО пересоздания виджетов на новом
        # прогоне, так что здесь менять session_state безопасно.
        runpod.terminate_pod(st.session_state.pod_id)
        st.session_state.pod_id = ''

    st.button("🛑 Остановить и удалить сервер (Остановить списание средств)", on_click=_stop_pod)
