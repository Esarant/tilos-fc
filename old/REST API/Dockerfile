FROM python:3.9-slim




RUN pip3 install --upgrade pip

COPY requirements.txt ./requirements.txt
COPY utils ./utils

COPY load_forecasting_server.py ./
COPY model.proto model_pb2.py model_pb2_grpc.py ./


WORKDIR /

RUN pip3 install -r requirements.txt

EXPOSE 8061 8062

ENTRYPOINT [ "python3","load_forecasting_server.py" ]
