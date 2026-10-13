FROM python:3.11-slim

LABEL maintainer="azahir@pa.uc3m.es"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# iw: the only system dependency -- this service reads 802.11 link stats
# via `iw dev <iface> link` / `station get`, nothing else.
RUN apt-get update && \
    apt-get install -y --no-install-recommends iw && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

ENTRYPOINT ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6002", "--workers", "1"]
