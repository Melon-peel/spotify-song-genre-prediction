# spotify-song-genre-prediction


## Навигация

* [Описание проекта](#description)
* [Установка](#installation)
* [TL;DR](#tldr)
* [Использование: обучение](#use_learning)
  + [Базовый сценарий (без логирования)](#use_learning_base)
  + [Запуск с логированием](#use_learning_logging)
  + [Запуск с логированием по кастомному адресу mlflow](#use_learning_logging_custom)
* [Использование: инференс](#use_infer)
  + [Базовый сценарий](#use_infer_base)
* [Настройка verbosity](#verbosity)

---
<a name='description'></a>
## Описание проекта

Задача данного проекта - предсказание жанра трека, используя информацию о
соответствующих ему аудио-фичах.

Аудио-фичи извлечены, используя Spotify API. Информация о жанрах извлечена из
сервиса EveryNoise.

**NOTE**:
На данный момент качество модели отвратительное, в основном поскольку:

- используемых признаков не хватает для адекватного предсказания (to be fixed)
- при 23 используемых признаках, на каждый класс (жанр) из примерно 1300
  приходится **всего** 100 объектов (to be fixed (or not))

---

<a name='installation'></a>
## Установка

- `git clone https://github.com/Melon-peel/spotify-song-genre-prediction.git`
- `python -m venv path/to/your/venv`
- `source path/to/your/venv/bin/activate`
- `cd spotify-song-genre-prediction`
- `poetry install`
- `pre-commit install`

---

<a name='tldr'></a>
## TL;DR 

### Для запуска без логирования:

`python train.py` 
`python infer.py`

### Для запуска с логированием:

`python train.py --logging=True` 
`python infer.py`

---

<a name='use_learning'></a>
## Использование: обучение

<a name='use_learning_base'></a>
### Базовый сценарий (без логирования)

`python train.py`:

- загружает данные и сохраняет их локально
- обучает модель и сохраняет её локально

**Notes**:

- для переобучения модели со своими данными используйте флаг `--search=local`

<a name='use_learning_logging'></a>
### Запуск с логированием

`python train.py --logging=True`:

- загружает данные и сохраняет их локально
- обучает модель и сохраняет её локально
- запускает логирование с помощью mlflow

**Notes**:

- для переобучения модели со своими данными используйте флаг `--search=local`
  (скрипт будет искать `data/train_test/train.csv` и `data/train_test/test.csv`)
- предполагается, что сервер запущен по адресу **http://128.0.1.1:8080**

<a name='use_learning_logging_custom'></a>
### Запуск с логированием по кастомному адресу mlflow

`python train.py --logging=True --host=host_address --port=port_address`:

- загружает данные и сохраняет их локально
- обучает модель и сохраняет её локально
- запускает логирование с помощью mlflow по адресу
  **http://\*host_address\*:\*port_address\***

**Notes**:

- для переобучения модели со своими данными используйте флаг `--search=local`

---

<a name='use_infer'></a>
## Использование: инференс

Запуск предполагает, что в `data/train_test/` есть файлы `train.csv` и
`test.csv`, а в `models/` есть файл `dt_clf.skops`. Если `python train.py` был
успешно выполнен до запуска `python infer.py`, файлы с данными и моделью уже
содержатся в соответствующих директориях

<a name='use_infer_base'></a>
### Базовый сценарий

`python infer.py`

- загружает данные и модель
- предсказывает классы объектов
- сохраняет предсказания в текущей директории в predictions.csv в формате
  `genre_actual,genre_predicted`

---
<a name='verbosity'></a>
## Настройка verbosity

`train.py` и `infer.py` запускаются по умолчанию в режиме verbose. Для изменения
поведения необходимо выставить флаг `--verbose=False`. Пример:
`python train.py --verbose=False`
