Харнесс OCR / Harness OCR — передача на Linux + NVIDIA GPU
Версия исходников: 0.1.0rc4

СТАТУС

Это единый исполняемый release candidate, а не уже аттестованный продовый движок.
Программные контракты и настоящая CUDA-сборка с offline imports проверены
в Linux CI. Фактическая работа Xiaomi/Qwen на RTX, задержки, VRAM и качество
на новых примерах подтверждаются отдельно на целевой машине. Упаковка не переносит
результаты старых экспериментов в обещания качества новой версии.

Состав: HTTP API и постоянные задания; контролируемый один процесс моделей;
Xiaomi-OCR-0; альтернативное чтение кириллицы; геометрия и доказательства полей;
Qwen 0.8B или 4B как выборщик подтверждённых кандидатов; ограниченное перечитывание;
проверки, review и трассы; адаптивный контроллер; раздельные исследовательские
инструменты. Корректные human-oracle данные никогда не включаются в обычный API.

В source ZIP отсутствуют веса, .env, ноутбучные окружения, сканы, журналы запуска
и персональные документы. Внутри source-manifest.json — перечень исходников и SHA.
Все тестовые протокольные примеры вымышленные и не являются OCR-бенчмарком.

ТРЕБОВАНИЯ К МАШИНЕ

Linux x86_64, Docker Engine и Compose v2 с NVIDIA Container Toolkit.
Целевой GPU: RTX PRO 5000 Blackwell, compute capability 12.0, 48 GB.
Для этого кандидата выбран консервативный release gate драйвера >=595.71.05.
Это версия драйвера CUDA 13.2 Update 2, а не общий минимум совместимости CUDA13.
Начальный ориентир — от32GB системной RAM и50GB свободного диска; реальные пики
и пределы размера документов измеряются на целевой конфигурации.

Официально проверены наличие тегов/колёс, а не их запуск на нашем GPU:
  python:3.12.16-slim-bookworm
  torch==2.15.0+cu132
  torchvision==0.29.2+cu132
  https://download.pytorch.org/whl/cu132
cu132 обозначает ветку CUDA13.2. Микроверсия библиотек определяется фактически
установленными зависимостями, а не названием индекса; doctor записывает их версии.
Базовый linux/amd64 manifest закреплён реальным digest:
  sha256:0f086103b026048cc50cb5da25ec435c023c5241a83fc44daea593377abc61c4
Индекс tag и платформенный manifest получены из официального Docker Registry v2;
SHA пересчитаны по сырым ответам. Метаданные — deployment/base-image-lock.json.
Docker build и offline imports реально выполнены в GitHub Actions; это не
GPU-инференс. Транзитивные зависимости не закреплены полным Linux wheels/hash lock; build freeze только описывает
один установленный результат и сам по себе не гарантирует повторяемость сборки.

Официальные источники:
  https://raw.githubusercontent.com/docker-library/official-images/master/library/python
  https://download.pytorch.org/whl/cu132/torch/
  https://download.pytorch.org/whl/cu132/torchvision/
  https://developer.nvidia.com/cuda/gpus
  https://docs.nvidia.com/cuda/archive/13.2.2/cuda-toolkit-release-notes/index.html
  https://docs.docker.com/compose/how-tos/gpu-support/
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

СБОРКА И ЗАГРУЗКА ВЕСОВ (BASH, ИЗ КОРНЯ РАСПАКОВАННОГО ПРОЕКТА)

  nvidia-smi
  docker compose version
  export HARNESS_API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  export HARNESS_REVIEW_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  export HARNESS_GPU_ID=0
  mkdir -p models
  docker compose build --pull

Токены разные, обязательны при запуске, передаются через среду; не добавляйте их
в Git/архив или в сообщения. Для перезапуска сохраните их в своём secret manager.
Фактический build может выявить несовместимость зависимости; pip check включён.

Подготовка скачивает только файлы конкретных ревизий из model_lock.json и проверяет
SHA каждого файла; это отдельная явная операция, не загрузка внутри запроса OCR:

  docker compose run --rm --no-deps --user "$(id -u):$(id -g)" \
    -v "$PWD/models:/models:rw" --entrypoint python ocr \
    -m harness_ocr.cli prepare-models --models /models --download

Все пять компонентов нужны для полного сравнения 0.8B/4B. Для переноса уже
скачанных весов используйте --local-sources JSONpath: объект
{"xiaomi":"/path/to/snapshot", "qwen-0.8b":"...", "qwen-4b":"...", ...}.
Пути относятся к контейнеру и должны быть явно примонтированы read-only.
В архиве лежат только манифесты, не13GB весов.

ЗАПУСК И ДИАГНОСТИКА

  docker compose up -d
  docker compose logs --tail 100 ocr
  docker compose exec ocr python scripts/doctor.py --models /models \
    --base-url http://127.0.0.1:8080 --output /reports/doctor-001.json

doctor проверяет Linux/Python, реальные версии torch и CUDA, BF16 kernel на GPU,
драйвер, SHA весов, pip check и readiness HTTP. Он не загружает модели и не
доказывает качество распознавания. Без --base-url готовность API отмечается
pending. Коды возврата:0 pass,1 fail,2 pending. JSON-файлы create-only: при
повторе используйте новое имя. Просмотр результата:

  docker compose cp ocr:/reports/doctor-001.json ./doctor-001.json

API слушает только localhost:8080; inference работает без скачиваний, с одним
worker и read-only моделями. Модели не запускаются параллельно. Runtime данные
в named volume harness_data, отчёты — harness_reports. Это защищённые данные:
трассы и принятые документы могут содержать персональную информацию. Встроенная
схема однопользовательская; multi-tenant RBAC/квоты/хранение требуют отдельного
контракта. Интернет-прокси, SSO, TLS и политика удаления относятся к окружению.

После независимого аудита Docker HEALTHCHECK проверяет authenticated readiness
(ready=true, fixture=false), а не только liveness. На холодном старте предусмотрено
180секунд start-period. Если процесс жив, но веса/инференс не готовы, контейнер
будет unhealthy; статус и причина доступны через doctor/readiness.

API и offline CLI используют общий HARNESS_LOCK_DIR=/var/lib/harness/locks.
Воркер держит compute lease, поэтому после аварии scheduler новый процесс не
запустит модели поверх ещё работающего осиротевшего воркера. Для внешнего процесса
или другого контейнера на той же GPU примонтируйте тот же каталог leases: разные
файловые namespaces не могут согласовать OS-lock автоматически. Подробности —
docs/deployment-readiness.md.

MCP-адаптер включён в полный Docker install. Подключение к вашему каталогу — stdio
командой из работающего контейнера:
  docker compose exec -T ocr python -m harness_ocr.cli mcp
Он обращается к тому же authenticated HTTP API с analysis token. Инструменты:
ocr_profiles, ocr_submit, ocr_status, ocr_result, ocr_cancel. Файл сначала загружается
через POST /v1/artifacts; модель получает opaque artifact_id. Через MCP не выдаются
произвольные пути, замена политик/весов и полномочия review/одобрения госуслуги.

ПРОВЕРКА HTTP И РЕАЛЬНОГО ИНФЕРЕНСА

Сначала получите реальные profile_id через CLI profiles / GET /v1/profiles.
Примонтируйте контролируемый файл, разрешённый для тестирования:

  docker compose run --rm --no-deps \
    -v "$PWD/test-documents:/test-documents:ro" --entrypoint python ocr \
    scripts/acceptance.py --base-url http://ocr:8080 \
    --document /test-documents/example.png --profile YOUR_PROFILE_ID \
    --qwen qwen-0.8b --output /reports/api-001.json

Проверяются health, запрет без токена, SHA загрузки, очередь/завершение, result,
trace и отмена отдельного задания. Отсутствующая ручная review-проверка даёт
pending, даже если инференс завершился. Полный тест дополнительно требует
--review-json /test-documents/review.json, содержащий реальную авторизованную
правку данного результата:

  {"field_key":"actual.field", "raw_value":"literally observed text",
   "evidence_ids":["real-existing-region-id"], "expected_revision":0}

Номер ревизии и IDs берите из фактических result/trace; не придумывайте их.
Скрипт проверяет запрет review для analysis token, успешную правку review token
и отказ при повторе со старой ревизией. Автоматическая acceptance не предлагает
значений полей и не подменяет ручную аннотацию. Гонка «завершилось до отмены»
записывается pending, а не как доказанная работоспособность отмены.

Для воспроизводимой проверки только lifecycle вместо --review-json можно явно
указать --review-observed-field passport.number. Скрипт возьмёт ровно текущее
raw_value, его реальные evidence_ids и revision из только что созданного result
и повторно подтвердит тот же литерал. Если поле не прочитано, статус pending.
Это проверка полномочий и revision CAS, не независимая human-аннотация качества.

СРАВНЕНИЕ СВЯЗОК НА GPU

Остановите API перед offline benchmark — у GPU должен быть один владелец.
Не запускайте doctor/inference/benchmark параллельно ради скорости.

  docker compose stop ocr

Для первого запуска без внешнего корпуса можно создать русский DEV smoke:

  docker compose run --rm --no-deps --entrypoint python ocr \
    scripts/make_smoke_corpus.py --output /reports/synthetic-dev-001
  docker compose run --rm --no-deps --entrypoint python ocr \
    scripts/gpu_acceptance.py --manifest /reports/synthetic-dev-001/manifest.json \
    --models /models --output /reports/smoke-bench-001 --seconds 3600 \
    --report /reports/smoke-gpu-acceptance-001.json

Это один вымышленный исходник с двумя коррелированными вариантами, а не пилот
на реальных паспортах. Даже идеальные ответы не квалифицируют профиль.

Подготовьте отдельный разрешённый dev либо blind manifest в benchmark-data:
  {"schema_version":"harness-bench/1", "split":"blind",
   "documents":[{"id":"doc-001", "original_id":"original-001",
     "path":"/benchmark-data/doc-001.png", "profile_id":"YOUR_PROFILE_ID",
     "gold":{"actual.field":"literal expected text"}}]}

Это эталон для scorer, не данные модели. Варианты одного исходника должны иметь
один original_id; на действительно независимых документах — разные IDs.
Не смешивайте DEV с blind и не меняйте эталоны после просмотра ошибок.

  docker compose run --rm --no-deps \
    -v "$PWD/benchmark-data:/benchmark-data:ro" --entrypoint python ocr \
    scripts/gpu_acceptance.py --manifest /benchmark-data/manifest.json \
    --models /models --output /reports/bench-001 --seconds 14400 \
    --report /reports/gpu-acceptance-001.json

Запускается один sequential bench, четыре arm:
A0 rules; A1 hybrid Qwen0.8B base; A3 Qwen4B base на matched OCR; A4 адаптивная стратегия.
Обе Qwen здесь — официальные base-веса с точными SHA из model lock. Математический
weight-edit предыдущего исследования не применён скрыто к OCR и не переносился
на4B или в непроверенный CUDA runtime. Отдельный будущий A2 проверяет edited0.8B
после фиксации patch/base SHA, воспроизводимого CUDA-переноса и независимого
OCR-сравнения. Сейчас таких результатов нет; гибридное ядро Xiaomi+Qwen работает
с исходными моделями и не меняет веса в ходе запросов.
wrapper проверяет, что есть все arms, исходные группы, все запланированные строки
и завершённые генерации. Частичные/ошибочные результаты не становятся pass.
Фактические артефакты bench сохраняются в output/benchmark; SHA начальной оптики
для каждой пары перепроверяются по сохранённым JSON, а fixture-запуски отвергаются.
CLI bench хранит сырые наблюдения и метрики полей/документов, время и VRAM.
Небольшой dev-набор и successful execution не дают production qualification:
она остаётся false, а release_status pending до независимых quality/risk gates.

РЕГИСТРАЦИЯ СБОРКИ И ПЕРЕДАЧА

После успешного build и запуска на целевой машине сохраните:

  docker image inspect harness-ocr:0.1.0rc4 --format '{{.Id}}'
  docker buildx imagetools inspect python:3.12.16-slim-bookworm@sha256:0f086103b026048cc50cb5da25ec435c023c5241a83fc44daea593377abc61c4
  docker compose run --rm --no-deps --entrypoint cat ocr /opt/harness-build.freeze.txt

Фиксируйте реальный final image ID, базовый manifest и установленные версии;
финальный registry digest появляется только после явной публикации образа.
Базовый image@sha256 уже закреплён, но для повторяемой поставки нужны также
полный wheels/hash lock и финальный проверенный image artifact после приёмки.

Source ZIP можно пересобрать без окружений/секретов/документов:
  python3 scripts/build_release.py --output ./harness-ocr-source-new.zip

Он создаёт SHA inventory и отказывается перезаписывать существующий архив.
Правила включения — allowlist, а не упаковка всей рабочей папки. Крупные модели
передаются отдельно по pinned model lock. Не отправляйте var/, reports/, .env,
сканы или защищённые трассы вместе с исходниками.

ЧТО ЕЩЁ НУЖНО ДО ПРОДОВОГО СТАТУСА

1. Реальный Docker build и BF16/model smoke на RTX, измерения p50/p95/VRAM/OOM.
2. Независимые реальные русские документы каждого профиля; ошибки и abstain,
   критические поля, источники/варианты и применимость полей считаются отдельно.
3. Отдельная calibration для auto-accept; fail-closed/review по умолчанию.
4. Нагрузочная/восстановительная проверка очереди, отмены, рестартов, PDF и лимитов.
5. Полный Linux lock с hash, проверенный image digest, аудит зависимостей,
   эксплуатационные retention/access/monitoring соглашения вашей инфраструктуры.

Не обещаем «покрытие30–50документов» только на основании наличия контрактов.
Область профилей и реально измеренное качество — разные артефакты.

GITHUB CI

.github/workflows/ci.yml запускает software-contracts на Ubuntu/Python3.12 при
push/PR: установка .[inference,mcp], pip check, unittest, проверка source ZIP.
Он не скачивает веса и не требует GPU. Реальный build CUDA Docker image
автоматически запускается после успешных software-contracts для PR и push в main;
также доступен ручной запуск с docker_build=true. Отдельный job собирает Dockerfile
и выполняет offline imports внутри образа, но не подменяет GPU doctor, генерацию
моделей или независимую оценку качества. Точные SHA официальных Actions получены
из GitHub API и сохранены в deployment/actions-lock.json.

СЛЕПОЕ ВИЗУАЛЬНОЕ ЧТЕНИЕ И МЧД В RC3

Compose включает HARNESS_VISUAL_VERIFIER=1. После ограниченных перечитываний Qwen
читает до двух исходных ROI без предлагаемого значения; сервер сравнивает полное
чтение дословно. Неполный охват, расхождение или abstain оставляют review.
Детерминированный A0 и остальные A1/A3/A4 явно не включают vision: его проверяют
отдельно. Перед запуском диагностики остановите API:

  docker compose stop ocr
  docker compose run --rm --no-deps --entrypoint python ocr \
    /app/scripts/visual_acceptance.py --models /models \
    --output /reports/visual-target-001 --seconds 600
  docker compose up -d

Это синтетическая функциональная проверка обоих Qwen, не оценка точности реальных
документов. Полные условия и поля отчёта: docs/visual-verifier.md.
МЧД003/EMCHD_1 обрабатывается native-адаптером без OCR/Qwen, только в описанной
области обычной ненотариальной доверенности: docs/mchd_native.md.
Подпись, отзыв и юридические полномочия остаются not_verified. Составные поля,
банковские ключи и неизвестные версии описаны в docs/typed-field-contracts.md.

СВОИ ДОКУМЕНТЫ БЕЗ ПОДГОТОВКИ GOLD

Краткий основной путь передачи: docs/colleague-handoff.md.
Перечень профилей, точные поля, форматы и лимиты: docs/supported-documents.md.
Шаблон входов: docs/own-documents.example.json. Последовательный клиент
scripts/run_documents.py использует уже работающий Compose API, сохраняет
result.json/trace.json/status.json каждого документа и общий summary.json.
Он не загружает вторую модель и не требует банковских/ведомственных интеграций.
Готовность к такому запуску и точность OCR — разные показатели отчёта.
