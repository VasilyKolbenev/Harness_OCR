Харнесс OCR / Harness OCR 0.1.0rc3

Начните с README_LINUX_GPU.txt: там точные команды сборки Docker, подготовки
весов, запуска API, doctor, lifecycle acceptance и сравнения0.8B/4B на GPU.

Что передаётся: единое ядро с доказательствами происхождения полей, версионированием
состояния, ограниченным перечитыванием, Qwen-выборщиком кандидатов, review, трассами
и контроллерами качества/стратегий. API и offline benchmark используют одно ядро.
Исследовательские F04/F05/F09 имеют отдельные JSON-контракты; примеры:
  src/harness_ocr/research/README.txt

Начальное сравнение использует официальные base-веса Qwen0.8B и4B из model lock.
Математический weight-edit из предыдущего исследования Qwen не применён скрыто
к OCR и не переносился на4B. Проверка патча — отдельный будущий arm A2 с собственным
SHA, проверкой CUDA-переноса и OCR-метриками; в этой поставке он не измерялся.

Основные команды:
  python -m harness_ocr.cli profiles
  python -m harness_ocr.cli prepare-models --help
  python -m harness_ocr.cli infer --help
  python -m harness_ocr.cli bench --help
  python scripts/doctor.py --help
  python scripts/acceptance.py --help
  python scripts/gpu_acceptance.py --help

Модели не входят в исходники. Их точные ревизии, имена файлов, размеры и SHA256:
  src/harness_ocr/data/model_lock.json
Базовый Linux image manifest закреплён в Dockerfile и deployment/base-image-lock.json.
Лицензия нового кода пока не выбрана; условия сторонних компонентов отдельны:
  LICENSE.txt
  THIRD_PARTY_NOTICES.txt

Новые контракты rc3: typed-fields-v1 для семи составных полей; native МЧД003
с оригинальным XSD и source replay; blind Qwen vision для исходных ROI.
Область применения и оставшиеся проверки:
  docs/typed-field-contracts.md
  docs/mchd_native.md
  docs/visual-verifier.md

Статус: release candidate, программные контракты тестируются локально. Docker/GPU
приёмка и независимые метрики OCR остаются отдельными проверками. Значения из gold
не используются в production inference; автоматическое принятие полей по умолчанию
закрыто до квалификации профиля и политики риска. Само наличие10профилей не означает
доказанного качества на30–50типах документов.

Тесты без нейронного инференса:
  python -m unittest discover -s tests -v

Сборка чистого исходного архива:
  python scripts/build_release.py --output /path/to/new-harness-ocr-source.zip
Архив исключает окружения, веса, .env, сканы, результаты и защищённые трассы.
