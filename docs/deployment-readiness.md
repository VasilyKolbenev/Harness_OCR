# Проверка поставки на Linux/GPU

Состояние RC6: **программный кандидат; GPU-приёмка требуется на целевой машине**. Software tests, CUDA build и небольшой предыдущий CPU-набор подтверждают разные свойства. Они не доказывают точность новых документов, GPU-инференс или производственную квалификацию. Основной поставочный путь — `bash scripts/gpu_handoff.sh`; HTTP-сервис для него не нужен. На хосте требуется Python 3.12+ без библиотек моделей, внутри образа используется закреплённый Python 3.12.

| Проверка | Что проверяется | Что это не подтверждает |
|---|---|---|
| `python -m unittest discover -s tests -v` | Контракты ядра, схемы, HTTP/MCP, права, очереди, остановка и OS leases, research/control runner | Качество нейромоделей и GPU-инференс |
| Actions `software-contracts` | Те же тесты на чистом Ubuntu/Python 3.12; сборка исходного ZIP | CUDA image/GPU/точность |
| Actions `cuda-image-build`, PR, push в `main` или ручной `docker_build=true` | Реальная сборка Dockerfile и offline-импорты torch, torchvision, Qwen3.5, API/MCP внутри образа | Наличие GPU и генерация моделей |
| `scripts/doctor.py --offline` | Драйвер, настоящий BF16 kernel, версии и SHA весов; API явно пропущен | Распознавание страниц и HTTP readiness |
| `scripts/gpu_smoke.py` | Настоящие CUDA-вызовы Xiaomi, выбранных Qwen text/vision, DB/CTC и YOLO, provenance/EOS/cleanup | Точность профилей на независимых документах |
| `experiments/gpu_delivery` | Metadata plan, отдельные adaptive workers, caps, охват, ресурсы; со scalar gold — exact только размеченных полей | Production qualification и надёжность неразмеченных полей |
| `scripts/export_gpu_delivery.py` | Разрешённые агрегаты, HTML/CSV и публичный ZIP без приватных артефактов | Правильность распознанных значений |
| `scripts/acceptance.py` | Настоящий API job, result/trace, отмена, полномочия и CAS правок | Независимое качество извлечения |
| `scripts/gpu_acceptance.py` | Прежний A0/A1/A3/A4 frozen-optics протокол, отдельно от нового GPU delivery | Сравнение нового adaptive/rescue; production qualification |
| `scripts/visual_acceptance.py` | Последовательный реальный Qwen vision 0.8B/4B, SHA пикселей, EOS, дословное сравнение на одном capture | Точность на реальных документах и независимость двух Qwen-проверок |
| Независимый корпус и F08 qualification | Ошибки/отказы по полям и исходникам, политика принятия | Поддержка других макетов/версий вне измеренного scope |

Точные команды находятся в [README_LINUX_GPU.txt](../README_LINUX_GPU.txt). CI выполняется после публикации коммита в GitHub; наличие YAML само по себе не означает успешное выполнение CI.

## Исправления после независимого аудита

- API и offline benchmark раньше использовали разные `data_dir` и могли одновременно владеть одной GPU. Теперь admission lease находится в общем `HARNESS_LOCK_DIR`; воркер удерживает отдельный compute lease до завершения своего процесса. Новый владелец не получает вычислительный доступ при аварии предыдущего scheduler, пока его осиротевший воркер ещё существует.
- Неудача запуска процесса освобождает admission lease и IPC; следующий вызов может восстановиться. Ошибка callback отмены завершает активный воркер. Per-call NaN/Inf/нулевой/отрицательный timeout отклоняется. Бюджет включает bootstrap процесса.
- Docker `healthy` раньше означал только доступность `/health/live`. Теперь healthcheck с серверным токеном требует `ready=true` и `fixture=false`. Живой процесс, отсутствующие веса и fixture-режим не дают зелёный health status.
- В исходную поставку включены CI workflow, lock официальных Actions, документация и отключённый шаблон runtime policy. Конфигурация с вымышленной квалификацией не поставляется.

## Общий volume блокировок RC6

API и GPU delivery используют один внешний Docker volume, по умолчанию `harness-ocr-device-locks`. Его имя задаётся `HARNESS_LOCK_VOLUME`; у host entrypoint есть эквивалентный `--lock-volume`. Handoff создаёт volume автоматически. Для отдельного операторского запуска API подготовьте его явно:

```bash
export HARNESS_LOCK_VOLUME=harness-ocr-device-locks
docker volume create "$HARNESS_LOCK_VOLUME"
```

В `compose.yaml` тот же volume монтируется в `/var/lib/harness/locks`, и этому пути равен `HARNESS_LOCK_DIR`. В standalone handoff он монтируется в `/locks`. Разные container paths допустимы: должны совпадать Docker volume и физические inode lease-файлов. Другой Compose project также обязан использовать этот volume. Приватные data/report volumes не заменяют общий namespace.

Перед работой узкий root init без сети и GPU монтирует только `/locks`. Он проверяет пять обычных файлов — cpu/cuda admission и compute, а также `gpu-delivery.lock`, — устанавливает root ownership, режим файлов 0666 и sticky directory 1777. Init не обнуляет содержимое и не подменяет inode. Batch работает под UID/GID оператора, API — под настроенными `HARNESS_UID/HARNESS_GID`. Так разные пользователи открывают те же OS leases, а приватные документы 0600 не требуют изменения прав.

Smoke и сравнительный sweep удерживают coordination lease; каждый Supervisor использует admission/compute leases. Авария scheduler не позволяет новому worker считать живой compute lease свободным. Оставшийся на диске lock-файл сам по себе не означает занятую GPU: владение связано с открытым дескриптором. Не удаляйте volume или файлы ради обхода блокировки.

Host preflight отказывается продолжать при OCR-сервисе на пересекающейся или неопределённой GPU, включая другой Compose project и старый namespace. Он показывает команду для штатной остановки, но не выполняет её. Доказанно раздельные explicit GPU назначения можно не блокировать. Совместный namespace остаётся обязательным; изменение имени volume не является способом получить параллелизм.

Для прямого запуска вне Docker нужен один общий абсолютный `HARNESS_LOCK_DIR` у всех владельцев устройства. Временный каталог по умолчанию не связывает автоматически разных пользователей или контейнеры. Это локальные OS leases, а не распределённый планировщик между узлами.

## Данные, права и измерения

Handoff использует host bind mounts для моделей, документов и отчётов, сохраняет текущий UID/GID, не копирует оригиналы и не меняет их права. Загрузка закреплённых моделей — отдельная сетевая стадия. Doctor, smoke и обработка документов выполняются с отключённой сетью и read-only моделями. Только временные smoke/comparison services используют host PID namespace для сопоставления собственных psutil PID с NVML; основной API остаётся изолированным. Неизвестный VRAM не заменяется нулём.

Публично передаётся только `RETURN_TO_TEAM.zip`: `report.html`, `metrics.csv`, `summary-public.json` (`harness-gpu-return/1`) и SHA manifest (`harness-gpu-return-manifest/1`). Полные rows/state/protocol/logs, имена файлов, gold и `private-input-map.json` остаются приватными. Без gold отчёт не утверждает измеренную точность; отсутствующая разметка не становится ошибкой модели.

HTTP/MCP — отдельный операторский режим. Для него нужны два разных токена, чтение model cache и запись в существующие state/report volumes под выбранным `HARNESS_UID/HARNESS_GID`. Изменение UID не исправляет ownership уже созданных named volumes автоматически. Однокомандный batch-путь не требует этого сервиса и не выполняет миграцию его данных.

GPU inference, VRAM, p50/p95, OOM/recovery и независимое качество остаются `pending`, пока нет отчётов с целевой машины и конкретного коммита/образа. Нельзя переносить CPU результат в GPU/BF16 как доказательство одинакового качества.
