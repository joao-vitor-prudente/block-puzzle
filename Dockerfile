FROM python3.11-alpine

WORKDIR /block_puzzle

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "src/main.py"]