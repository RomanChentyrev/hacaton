**AIJ Contest 2025**
===

Запуск бейзлайна, пример сабмита.

**Настройка переменных окружения**

Создайте `.env` файл в корневом каталоге, он должен содержать следующие обязательные значения. 
GigaChat используется в качестве основной LLM для прогона тестов.

```bash
GIGACHAT_TOKEN = "<GigaChat Token>"
GIGACHAT_SCOPE = "<GigaChat Scope>"
```

**Установка зависимостей**
---
```bash
cd submisson
pip install -r requirements.txt
```

**Загрузка данных**
Скачайте со страницы соревнования файлы `user_history.parquet`, `facts_db.parquet`, путь до этих файлов укажите в скрипте `test_baseline.py`


### Проверка бейзлайна
```bash
python test_baseline.py
```
