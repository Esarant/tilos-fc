
from concurrent import futures
import time
import logging
# import the generated classes :
import model_pb2
import model_pb2_grpc
import grpc
import pandas as pd
import json
import os
# import the function we made :
import utils.load_forecasting_functions as load_forecasting_functions

port = 8061

# create a class to define the server functions, derived from

class PredictServicer(model_pb2_grpc.PredictServicer):
    def  LoadForecasting(self, request, context):
        logging.info(f'Large Scale Load Forecasting at an Island Level service called with request { request }')
        
        
        try:
       
            model_input = json.loads(request.input_message)
     
        except Exception as e:
            logging.error('error occured while accessing request',e)
            context.set_details('please verify that the input is valid')
            #context.set_code(grpc.StatusCodes.INVALID_ARGUMENT)
            return model_pb2.OutputMessage()

        logging.info(f'model input is: {model_input}')

        response = load_forecasting_functions.decision_tree_rest(model_input)
        logging.info(response)
        predict = model_pb2.OutputMessage()

        
        predict.output_message = json.dumps(response)

        logging.info(f'response from model is {response}')

        return predict

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_PredictServicer_to_server(PredictServicer(), server)
    server.add_insecure_port("[::]:{}".format(port))
    server.start()
    logging.info("Large Scale Load Forecasting at an Island Level server starts listening at " + str(port))
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    #with open("config.yaml", "r") as ymlfile:
    #    config = yaml.safe_load(ymlfile)
    serve()
