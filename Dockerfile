from python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip3 install Cython
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

COPY backends/ backends/
COPY models/ models/
COPY server.py server.py
COPY BufferedInput.py BufferedInput.py