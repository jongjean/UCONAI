FROM python:3.10-slim
WORKDIR /app
COPY api_server.py ./
RUN pip install --no-cache-dir fastapi uvicorn ultralytics
EXPOSE 4000
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "4000"]
