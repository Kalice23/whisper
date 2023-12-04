from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text
from pywhispercpp.model import Model

# Указываем путь до файла с конфигом, он должен быть в той же директории, как сказано на шаге 3.
pipeline = Pipeline.from_pretrained("config.yaml")

# Указываем название модели large и путь до директории с whisper-моделями из шага 1.
model = Model('large', '/whisper.cpp/models', n_threads=6)

# Указываем путь до аудио-файл, кторый будем расшифровывать в текст. Путь обязательно абсолютный.
while True:
    file_name = input("Укажите путь до файла: ")
    asr_result = model.transcribe(file_name, language="ru")

# Конвертация результата в формат, который понимает pyannote-whisper.
    result = {'segments': list()}

    for item in asr_result:
        result['segments'].append({
            'start': item.t0 / 100,
            'end': item.t1 / 100,
            'text': item.text
            }
        )

    # Сегментация аудио-файла на реплики спикеров. Путь обязательно абсолютный.
    diarization_result = pipeline(file_name)

    # Пересечение расшифровки и сегментаци.
    final_result = diarize_text(result, diarization_result)

    # Вывод результата.
    for seg, spk, sent in final_result:
        line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}'
        print(line)
