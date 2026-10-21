# Overfit Court web app. Runs anywhere that serves a container on a port,
# including Nebius Serverless Endpoints (CPU is plenty: the models run on
# Token Factory and the backtests run in Token Factory Sandboxes).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OC_EXECUTOR=nebius STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY overfit_court ./overfit_court
COPY app.py .

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
