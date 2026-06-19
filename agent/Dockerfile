# Huginn production image
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies for scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY huginn ./huginn

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e ".[all]"

# Runtime stage
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create a non-root user for running the agent
RUN groupadd -r huginn && useradd -r -g huginn -m -d /home/huginn huginn

# Copy installed site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Runtime state directories
RUN mkdir -p /app/.huginn /app/workspace \
    && chown -R huginn:huginn /app

USER huginn

ENV HUGINN_WORKSPACE=/app/workspace
ENV HUGINN_PROMPT_CACHE_CONTROL=0
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "huginn.server"]
