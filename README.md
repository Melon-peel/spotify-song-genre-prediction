# spotify-song-genre-prediction

Задача данного проекта - предсказание жанра трека, используя информацию о
соответствующих ему аудио-фичах.

Аудио-фичи извлечены, используя Spotify API. Информация о жанрах извлечена из
сервиса EveryNoise.

### NOTE:

На данный момент качество модели отвратительное, в основном поскольку:

- используемых признаков не хватает для адекватного предсказания (to be fixed)
- при 23 используемых признаках, на каждый класс (жанр) из примерно 1300
  приходится **всего** 100 объектов (to be fixed (or not))

---

### Установка

- `git clone https://github.com/Melon-peel/spotify-song-genre-prediction.git`
- `python -m venv path/to/your/venv`
- `source path/to/your/venv/bin/activate`
- `cd spotify-song-genre-prediction`
- `poetry install`
- `pre-commit install`

---

### TL;DR

#### Для запуска без логгирования:

`python train.py` `python infer.py`

#### Для запуска с логгированием:

`python train.py --logging=True` `python infer.py`

---

### Использование: обучение

#### Базовый сценарий (без логгирования)

`python train.py`:

- загружает данные и сохраняет их локально
- обучает модель и сохраняет её локально

**Notes**:

- для переобучения модели со своими данными используйте флаг `--search=local`

#### Запуск с логгированием

`python train.py --logging=True`:

- загружает данные и сохраняет их локально
- обучает модель и сохраняет её локально
- запускает логгирование с помощью mlflow

**Notes**:

- для переобучения модели со своими данными используйте флаг `--search=local`
  (скрипт будет искать `data/train_test/train.csv` и `data/train_test/test.csv`)
- предполагается, что сервер запущен по адресу **http://128.0.1.1:8080**

#### Запуск с логгированием по кастомному адресу mlflow

`python train.py --logging=True --host=host_address --port=port_address`:

- загружает данные и сохраняет их локально
- обучает модель и сохраняет её локально
- запускает логгирование с помощью mlflow по адресу
  **http://\*host_address\*:\*port_address\***

**Notes**:

- для переобучения модели со своими данными используйте флаг `--search=local`

---

### Использование: инференс

Запуск предполагает, что в `data/train_test/` есть файлы `train.csv` и
`test.csv`, а в `models/` есть файл `dt_clf.skops`. Если `python train.py` был
успешно выполнен до запуска `python infer.py`, файлы с данными и моделью уже
содержатся в соответствующих директориях

#### Базовый сценария

`python infer.py`

- загружает данные и модель
- предсказывает классы объектов
- сохраняет предсказания в текущей директории в predictions.csv в формате
  `genre_actual,genre_predicted`

---

### Verbosity

`train.py` и `infer.py` запускаются по умолчанию в режиме verbose. Для изменения
поведения необходимо выставить флаг `--verbose=False`. Пример:
`python train.py --verbose=False`
