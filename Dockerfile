FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# libgomp1 is required by OpenCV/SciPy OpenMP threads on slim Debian images
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python3 -c "import cv2, numpy, scipy"

COPY solver.py .

ENTRYPOINT ["python3", "/app/solver.py"]
CMD ["/frames"]