FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY bike_demand_api ./bike_demand_api
COPY artifacts ./artifacts

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "bike_demand_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
