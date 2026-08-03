import streamlit as st
import runpod
import requests
import time
import base64
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

if st.button("🚀 Создать сервер, установить окружение и запустить"):
    with st.spinner("Запрос к RunPod..."):
        # Bash-скрипт с подробным логированием каждого шага (set -x) и отключенной буферизацией Python.
        # Переменная окружения PYTHONUNBUFFERED=1 теперь передается только внутри bash-скрипта.
        
        # ЧИСТЫЙ bash-скрипт без экранирования кавычек для передачи через Base64
        raw_script = """
        set -x
        echo "=== СТАРТ ИНИЦИАЛИЗАЦИИ ==="
        cd /workspace
        apt-get update
        apt-get install -y ffmpeg
        if [ ! -d /workspace/venv ]; then
            echo "=== СОЗДАНИЕ VENV И УСТАНОВКА ЗАВИСИМОСТЕЙ ==="
            python3 -m venv /workspace/venv
            /workspace/venv/bin/pip install fastapi uvicorn pydub demucs==4.1.0 transformers>=5.0.0 accelerate orjson requests
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
        echo "=== ЗАПУСК UVICORN ==="
        export PYTHONUNBUFFERED=1
        /workspace/venv/bin/python -m uvicorn backend_server:app --host 0.0.0.0 --port 5000
        """
        
        # Кодируем скрипт в Base64, чтобы обойти баг RunPod SDK с парсингом символов =, ", и пробелов
        encoded_script = base64.b64encode(raw_script.encode('utf-8')).decode('utf-8')
        
        # Передаем декодер прямо в docker_args
        startup_script = f"bash -c 'echo {encoded_script} | base64 -d | bash'"
        
        try:
            # Убран параметр env, так как он вызывал GraphQL Syntax Error: Expected Name, found "=" в RunPod API.
            # Буферизация уже отключена внутри startup_script через export.
            pod = runpod.create_pod(
                name="AiDubbing-GPU-Backend",
                image_name="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
                gpu_type_id="NVIDIA GeForce RTX 4090",
                cloud_type="SECURE",
                volume_in_gb=50,
                container_disk_in_gb=20,
                ports="5000/http",
                docker_args=startup_script
            )
            st.session_state['pod_id'] = pod['id']
            st.success(f"✅ Сервер создан! ID: {pod['id']}")
            st.info("⚠️ ВНИМАНИЕ: Зайдите в панель RunPod. Первые минуты смотрите в 'System Logs' (загрузка образа). Как только сервер запустится, процесс установки пакетов пойдет в 'Container Logs'.")
        except Exception as e:
            st.error(f"Ошибка при создании сервера: {e}")

# Поле для ID сервера (сохраняется в сессии)
pod_id = st.text_input("ID запущенного Pod'а (подставится автоматически):", value=st.session_state.get('pod_id', ''))

if pod_id:
    # URL для связи с портом 5000 внутри вашего Pod'а через прокси RunPod
    backend_url = f"https://{pod_id}-5000.proxy.runpod.net/dub"
    
    st.header("2. Загрузка и Дубляж")
    audio_file = st.file_uploader("Загрузите исходный файл (.aac, .mp3)")
    
    if audio_file is not None:
        if st.button("Инициировать дубляж"):
            with st.status("Оркестрация процесса...", expanded=True):
                st.write("1. Выполнение транскрипции (OpenAI Whisper)...")
                
                # Транскрипция с точными таймкодами фраз
                transcription_data = llm_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("audio.aac", audio_file.getvalue()),
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
                
                st.write("2. Адаптивный перевод (с учетом тайминга)...")
                payload_segments = []
                
                # В новой версии OpenAI SDK (v1.0+) объекты возвращаются как Pydantic модели, 
                # поэтому доступ к свойствам осуществляется через точку (segment.start), а не как к словарю.
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
                
                # Отправка задачи на наш FastAPI сервер внутри RunPod. 
                # Добавлен параметр timeout, чтобы запрос не завис навсегда, так как генерация аудио занимает время.
                try:
                    response = requests.post(
                        backend_url,
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
                        st.error(f"Сервер недоступен (Код {response.status_code}). Возможно, установка пакетов еще не завершилась. Подождите пару минут.")
                except requests.exceptions.Timeout:
                    st.error("Превышено время ожидания ответа от сервера. Проверьте логи RunPod.")

    st.markdown("---")
    if st.button("🛑 Остановить и удалить сервер (Остановить списание средств)"):
        runpod.terminate_pod(pod_id)
        st.session_state['pod_id'] = ''
        st.success("Сервер уничтожен.")
        st.rerun()
