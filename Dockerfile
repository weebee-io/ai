FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./finance-segmentation-api/app /app/app
COPY ./finance-segmentation-api/models /app/models
COPY gunicorn_conf.py /app/
COPY uvicorn_start.sh /app/

RUN chmod +x /app/uvicorn_start.sh

EXPOSE 8080

CMD ["./uvicorn_start.sh"]