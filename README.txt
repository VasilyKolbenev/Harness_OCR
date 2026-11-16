Харнесс OCR / Harness OCR 0.1.0rc6

Передача для GPU-тестов: начните с README_LINUX_GPU.txt.
  bash scripts/gpu_handoff.sh --input /absolute/path/to/documents --profile D07 --output "$PWD/reports/run-001"

Linux + Docker/Compose + NVIDIA Container Toolkit + host Python3.12+.
Цель: RTX PRO 5000 Blackwell48GB. Команда собирает образ, загружает проверенные
веса, выполняет настоящий GPU smoke и сравнивает adaptive DB с DB+YOLO-L
для Qwen0.8B и4B. Xiaomi, перечитывание и строгие проверки включены.
--include-yolo-x добавляет X; --manifest задаёт разные профили в одной пачке.
Модели не входят в ZIP. Профили/поля: docs/supported-documents.md.

Прислать: reports/run-001/RETURN_TO_TEAM.zip. Он содержит безопасные метрики,
HTML и CSV без сканов, значений полей, gold, приватных трасс и логов.
Для точности нужен отдельный --gold: формат в README_LINUX_GPU.txt.

Это release candidate для измерений. Выполнение/покрытие/формат/точность
учитываются отдельно. Неясность даёт review, автоматическое принятие закрыто
до квалификации. Документы не исправляются по догадке. Наличие10профилей
не доказывает покрытие30–50типов и всех макетов. Реальный GPU smoke является
обязательной проверкой на целевом компьютере.

Архитектура: docs/architecture.md; последовательность: docs/sequence.mmd.
Предыдущие CPU-измерения: docs/rc5-quality-findings.md.
F01–F09 сохранены; контракты: src/harness_ocr/research/README.txt.
Патч математической Qwen не переносился скрыто в OCR или на4B.
Ревизии/веса/SHA: src/harness_ocr/data/model_lock.json.
Лицензии: LICENSE.txt и THIRD_PARTY_NOTICES.txt.

Тесты без моделей: python -m unittest discover -s tests -v
Чистый исходный архив: python scripts/build_release.py --output /path/to/new-source.zip
