FROM python:3.9-slim

WORKDIR /app

# Cai dat dependencies: PyTorch CPU tu kho rieng, cac thu vien khac tu PyPI mac dinh
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy matplotlib


# Copy ma nguon train.py vao container
COPY train.py .

# Khoi chay chuong trinh mac dinh
CMD ["python", "train.py"]
