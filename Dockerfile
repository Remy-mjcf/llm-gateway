# Dockerfile: builds a small image that runs the llm-gateway HTTP API.
# Build:  docker build -t llm-gateway .
# Run:    docker run --rm -p 8000:8000 --env-file .env llm-gateway

# Slim Debian image with Python 3.12 (same minor version as the local venv)
FROM python:3.12-slim

# Don't write .pyc files into the image, and don't buffer stdout/stderr
# so log lines appear in `docker logs` immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create an unprivileged user so the server never runs as root inside the container
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Copy only what the build needs. pyproject.toml declares the dependencies,
# and hatchling also requires README.md because pyproject.toml lists it as the readme.
COPY pyproject.toml README.md ./
COPY src ./src

# Install the package and its runtime dependencies (no dev extras: no pytest, ruff, mypy).
# This is a regular install, not editable, so the image doesn't depend on the source layout.
RUN pip install .

# Drop root for everything that follows
USER appuser

# Document the port the server listens on (publishing it is done with `docker run -p`)
EXPOSE 8000

# Ask /health every 30s. It makes no provider calls, so it costs no API quota.
# python's urllib is used because slim images ship without curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# Start the API. 0.0.0.0 lets traffic from outside the container reach it.
# API keys are NOT baked into the image: they arrive as environment variables at run time.
CMD ["uvicorn", "llm_gateway.api:app", "--host", "0.0.0.0", "--port", "8000"]
