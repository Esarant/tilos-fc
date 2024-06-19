import grpc
import logging
import model_pb2
import model_pb2_grpc
import numpy as np
import pandas as pd
import json

port_address = 'localhost:8061'


def get_LoadForecasting(stub):
    params = {
"horizon": 3,
"data": [
{
"Temperature": 35.2,
"Cloud Coverage": 7.5,
"Radiation": 385,
"Month": "June",
"Hour": 11,
"Last_week_production": 2000,
"Last_week_average_production": 3561
},
{
"Temperature": 25.4,
"Cloud Coverage": 9.5,
"Radiation": 485,
"Month": "July",
"Hour": 12,
"Last_week_production": 3000,
"Last_week_average_production": 2834
},
{
"Temperature": 45.4,
"Cloud Coverage": 2.5,
"Radiation": 885,
"Month": "August",
"Hour": 10,
"Last_week_production": 4000,
"Last_week_average_production": 3834
}
]
}

    return stub.LoadForecasting(
        model_pb2.Input(
            input_message = json.dumps(params)
        )
    )


def run():
    with grpc.insecure_channel(port_address) as channel:

        stub = model_pb2_grpc.PredictStub(channel)
        try:
            response = get_LoadForecasting(stub)
        except grpc.RpcError as e:
            print(f"grpc error occurred: {e.details()}, {e.code().name}")
        except Exception as e:
            print(f"error occurred: {e}")
        else:
            output = json.loads(response.output_message)
            print(output)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    run()
