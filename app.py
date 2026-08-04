import streamlit as st
import runpod
import requests
import time
import base64
import os
import secrets as pysecrets
from openai import OpenAI

st.set_page_config(page_title="AiDubbing V4", layout="centered")
st.title("🎙️ AiDubbing V4: Управление и Дубляж")

# --- Ключи и токены -----------------------------------------------------
with st.expander("🔑 Настройки API ключей", expanded=True):
    openai_key = st.text_input("OpenAI API Key", type="password")
    runpod_key = st.text_input("RunPod API Key", type="password")

    # ИСПРАВЛЕНИЕ (безопасность): backend_server.py раньше вообще не проверял
    # Authorization на /dub, хотя app.py его отправлял. Значит, любой, кто
    # узнал URL пода (https://{pod_id}-5000.proxy.runpod.net/dub), мог
    # запускать дорогой GPU-инференс за ваш счёт без ключа.
    # Здесь генерируем отдельный секрет (не сам RUNPOD_API_KEY), прокидываем
    # его в под как переменную окружения при старте, и с этим же секретом
    # подписываем запросы к /dub.
    if "dub_auth_token" not in st.session_state:
        st.session_state.dub_auth_token = pysecrets.token_urlsafe(24)
    dub_auth_token = st.text_input(
        "Секрет для авторизации /dub (генерируется автоматически)",
        key="dub_auth_token",
        help="Этот же секрет будет передан в под как переменная окружения DUB_AUTH_TOKEN.",
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
        raw_script = f"""
        cd /workspace
        touch install.log

        python3 -m http.server 5000 &
        HTTP_PID=$!

        {{
            set -x
            echo "=== СТАРТ ИНИЦИАЛИЗАЦИИ ==="
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
            echo "=== УСТАНОВКА ЗАВЕРШЕНА ==="
        }} >> install.log 2>&1

        kill $HTTP_PID
        export PYTHONUNBUFFERED=1
        export DUB_AUTH_TOKEN='{dub_auth_token}'
        /workspace/venv/bin/python -m uvicorn backend_server:app --host 0.0.0.0 --port 5000
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
                ports="5000/http",
                docker_args=startup_script
            )
            st.session_state['pod_id'] = pod['id']
            st.success(f"✅ Сервер создан! ID: {pod['id']}")
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка при создании сервера: {e}")

# ИСПРАВЛЕНИЕ: раньше text_input получал одновременно value=... и не имел
# key=, поэтому после первого ручного редактирования поля новый pod_id из
# session_state переставал попадать в виджет при следующих rerun. Теперь
# виджет сам является хранилищем session_state.pod_id.
if "pod_id" not in st.session_state:
    st.session_state.pod_id = ""
pod_id = st.text_input("ID запущенного Pod'а (подставится автоматически):", key="pod_id")

if pod_id:
    backend_url = f"https://{pod_id}-5000.proxy.runpod.net"
    dub_endpoint = f"{backend_url}/dub"

    st.header("2. Статус установки сервера")

    # ИСПРАВЛЕНИЕ (UI): раньше логи показывались через st.code() без height,
    # то есть блок разъезжался на весь текст и не был ни зафиксирован по
    # высоте, ни независимо скроллируем — обновление логов требовало ручного
    # нажатия кнопки. Теперь это st.fragment с run_every: часть страницы
    # перерисовывается сама по таймеру, не трогая остальной интерфейс
    # (https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment),
    # а st.code(..., height=...) даёт фиксированную высоту и скролл внутри
    # блока, как окно консоли (height добавлен в st.code в PR streamlit#10080).
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
                if "=== УСТАНОВКА ЗАВЕРШЕНА ===" in text:
                    st.info("Пакеты установлены, FastAPI поднимается (модель грузится в VRAM, обычно ~30с).")
            else:
                docs_res = requests.get(f"{backend_url}/docs", timeout=5)
                if docs_res.status_code == 200:
                    st.code("✅ install.log больше не отдаётся — FastAPI поднялся, сервер готов принимать /dub.",
                            language="bash", height=400)
                    st.success("Сервер полностью готов к работе.")
                else:
                    st.code(f"Ожидание... код ответа install.log: {res.status_code}", language="bash", height=400)
        except requests.exceptions.RequestException:
            st.code("Контейнер ещё скачивается/инициализируется на физической ноде...", language="bash", height=400)

    log_console()

    st.header("3. Загрузка и Дубляж")
    audio_file = st.file_uploader("Загрузите исходный файл (.aac, .mp3, .m4a, .wav)")

    if audio_file is not None:
        if st.button("Инициировать дубляж"):
            with st.status("Оркестрация процесса...", expanded=True):
                st.write("1. Выполнение транскрипции (OpenAI Whisper)...")

                original_ext = os.path.splitext(audio_file.name)[1].lower()
                valid_extensions = ['.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.oga', '.ogg', '.wav', '.webm']
                safe_extension = original_ext if original_ext in valid_extensions else '.m4a'
                safe_filename = f"audio{safe_extension}"

                transcription_data = llm_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=(safe_filename, audio_file.getvalue()),
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
                encoded_audio = base64.b64encode(audio_file.getvalue()).decode('utf-8')

                try:
                    response = requests.post(
                        dub_endpoint,
                        json={"audio_base64": encoded_audio, "segments": payload_segments},
                        # ИСПРАВЛЕНИЕ: раньше сюда шёл runpod_key (ключ доступа
                        # ко всему аккаунту RunPod), хотя backend его всё
                        # равно не проверял. Теперь это отдельный секрет,
                        # который backend_server.py реально валидирует.
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
                        st.error("401: секрет DUB_AUTH_TOKEN не совпадает с тем, что задан в поде (пересоздайте под или сверьте значение).")
                    else:
                        st.error(f"Сервер недоступен (Код {response.status_code}). Убедитесь, что логи установки завершены.")
                except requests.exceptions.Timeout:
                    st.error("Превышено время ожидания ответа от сервера. Процесс генерации занимает много времени.")

    st.markdown("---")
    if st.button("🛑 Остановить и удалить сервер (Остановить списание средств)"):
        runpod.terminate_pod(pod_id)
        st.session_state['pod_id'] = ''
        st.success("Сервер уничтожен.")
        st.rerun()
