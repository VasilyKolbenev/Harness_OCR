# Candidate Linux amd64 build. Run scripts/doctor.py and GPU acceptance on target.
# Manifest SHA verified via Docker Registry v2; target build/runtime still pending.
ARG PYTHON_IMAGE=python:3.12.16-slim-bookworm@sha256:0f086103b026048cc50cb5da25ec435c023c5241a83fc44daea593377abc61c4
FROM ${PYTHON_IMAGE}
LABEL org.opencontainers.image.title="Harness OCR" \
      org.opencontainers.image.version="0.1.0rc3" \
      org.opencontainers.image.description="GPU release candidate; target acceptance required"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 HARNESS_DEVICE=cuda HARNESS_DTYPE=bfloat16 \
    HARNESS_DATA=/var/lib/harness HARNESS_MODELS=/models HARNESS_LOCK_DIR=/var/lib/harness/locks \
    OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
WORKDIR /app
RUN apt-get update && apt-get install --no-install-recommends -y \
      ca-certificates libglib2.0-0 libgomp1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home harness \
    && mkdir -p /models /var/lib/harness /reports \
    && chown -R harness:harness /var/lib/harness /reports
COPY requirements-cuda.txt /app/requirements-cuda.txt
RUN python -m pip install --no-cache-dir -r requirements-cuda.txt
COPY pyproject.toml requirements-inference.txt /app/
COPY LICENSE.txt THIRD_PARTY_NOTICES.txt /app/
COPY LICENSES/ /app/LICENSES/
COPY src/ /app/src/
COPY scripts/ /app/scripts/
RUN python -m pip install --no-cache-dir -r requirements-inference.txt \
    && python -m pip check \
    && python -m pip freeze > /opt/harness-build.freeze.txt
USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD ["python", "/app/scripts/healthcheck.py"]
CMD ["uvicorn", "harness_ocr.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
