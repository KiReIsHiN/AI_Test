import streamlit as st
import runpod
import requests
import time
import base64
import os
from openai import OpenAI

# Настройки страницы и инициализация
st.set_page_config(page_title="AiDubbing V4", layout="centered")
st.title("🎙️ AiDubbing V4: Управление и Дубляж")

# Безопасный ввод ключей
with st.expander("🔑 Настройки API ключей", expanded=True):
    openai_key = st.text_input("OpenAI API Key", type="password")
    runpod_key = st.text_input("RunPod API Key", type="password")

if not openai_key or not runpod_key:
    st.warning("Пожалуйста, введите оба ключа для начала работы.")
    st.stop()

runpod.api_key = runpod_key
llm_client = OpenAI(api_key=openai_key)

st.header("1. Управление GPU-сервером")

# Настройки сервера вынесены в UI для обхода нехватки инстансов
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
        # Обновленный скрипт:
        # 1. Запускает временный http-сервер на порту 5000 для трансляции логов.
        # 2. Перенаправляет весь поток установки в install.log.
        # 3. Исправляет ошибку с > в transformers через кавычки.
        # 4. Убивает временный сервер и запускает FastAPI.
        raw_script = """
        cd /workspace
        touch install.log
        
        # Запускаем временный сервер для отдачи логов в UI
        python3 -m http.server 5000 &
        HTTP_PID=$!
        
        # Весь вывод направляем в install.log
        {
            set -x
            echo "=== СТАРТ ИНИЦИАЛИЗАЦИИ ==="
            apt-get update
            apt-get install -y ffmpeg
            if [ ! -d /workspace/venv ]; then
                echo "=== СОЗДАНИЕ VENV И УСТАНОВКА ЗАВИСИМОСТЕЙ ==="
                python3 -m venv /workspace/venv
                # Кавычки вокруг transformers обязательны для защиты от парсера Bash
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
        } >> install.log 2>&1
        
        # Освобождаем порт 5000 и запускаем основной бэкенд
        kill $HTTP_PID
        export PYTHONUNBUFFERED=1
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

# Поле для ID сервера (сохраняется в сессии)
pod_id = st.text_input("ID запущенного Pod'а (подставится автоматически):", value=st.session_state.get('pod_id', ''))

if pod_id:
    backend_url = f"https://{pod_id}-5000.proxy.runpod.net"
    dub_endpoint = f"{backend_url}/dub"
    
    st.header("2. Статус установки сервера")
    
    # Блок мониторинга установки
    if st.button("🔄 Обновить логи установки"):
        # Добавляем параметр времени, чтобы избежать кэширования прокси-сервером RunPod
        log_url = f"{backend_url}/install.log?t={int(time.time())}"
        try:
            res = requests.get(log_url, timeout=5)
            if res.status_code == 200:
                st.code(res.text, language="bash")
                if "=== УСТАНОВКА ЗАВЕРШЕНА ===" in res.text:
                    st.success("Пакеты установлены! Запускается FastAPI (модели загружаются в видеокарту, это займет еще ~30 секунд).")
            else:
                # Если 404, значит временный сервер убит и работает FastAPI (маршрут /install.log не существует). 
                # Проверим доступность авто-документации FastAPI (/docs).
                docs_res = requests.get(f"{backend_url}/docs", timeout=5)
                if docs_res.status_code == 200:
                    st.success("✅ Сервер FastAPI полностью запущен и готов к приему аудио!")
                else:
                    st.warning(f"Ожидание запуска. Код ответа сервера: {res.status_code}.")
        except requests.exceptions.RequestException:
             st.info("Контейнер скачивается или инициализируется на физической ноде. Подождите пару минут и нажмите кнопку снова.")
    
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
                    if dur < 0.6: continue
                    
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
                        headers={"Authorization": f"Bearer {runpod_key}"},
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
