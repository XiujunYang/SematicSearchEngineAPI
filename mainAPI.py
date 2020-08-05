from flask import Flask, jsonify, request
from flask_restplus import Api, Resource, fields
import tensorflow_hub as hub
import numpy as np
import fasttext
import datetime
import sys

app = Flask(__name__)
app.config['RESTPLUS_VALIDATE'] = True
app.config['RESTPLUS_MASK_SWAGGER'] = False

api = Api(app=app)
ns = api.namespace('sentence_embedding', description='sentence embedding')

algorithms = {}
current_algorithm=None
model=None
# first argv is python file.
print('args are',str(sys.argv))

''' # rf
root@013b7fe9b384:/app# free -ht
              total        used        free      shared  buff/cache   available
Mem:          7.8Gi       1.7Gi       5.7Gi       0.0Ki       409Mi       5.8Gi
Swap:         1.0Gi       610Mi       413Mi
Total:        8.8Gi       2.3Gi       6.1Gi

# only fasttext_eng
root@6ab1149ab448:/app# free -ht
              total        used        free      shared  buff/cache   available
Mem:          7.8Gi       7.3Gi       119Mi       0.0Ki       397Mi       286Mi
Swap:         1.0Gi       610Mi       413Mi
Total:        8.8Gi       7.9Gi       532Mi
'''

#model_dir = '/Users/xiujun/codebase/python/SimilarAndSematicSearchEngine/data/algorithms/'
model_dir = '/app/models/'

# ----------------------------------------------------------------------------------------
# tf
# ----------------------------------------------------------------------------------------
#embed = hub.load(model_dir + 'tf/universal-sentence-encoder_4')
def load_tf_model(alg_name, path):
    print("load", alg_name, "path:", model_dir+path)
    #algorithms[alg_name]=hub.load(model_dir + path)
    return hub.load(model_dir + path)

# ----------------------------------------------------------------------------------------
# fasttext
# ----------------------------------------------------------------------------------------
#fasttext_eng = fasttext.load_model(model_dir + 'fasttext/cc.en.300.bin')
#fasttext_ara = fasttext.load_model(model_dir + 'fasttext/cc.ar.300.bin')
def load_fasttext_model(alg_name, path):
    print("load", alg_name, "path:", model_dir+path)
    #algorithms[alg_name]=fasttext.load_model(model_dir + path)
    return fasttext.load_model(model_dir + path)

# ----------------------------------------------------------------------------------------
# load model path
# ----------------------------------------------------------------------------------------
for alg in sys.argv[1].split(';'):
    alg_name = alg.split(':')[0].strip().lower()
    alg_path = alg.split(':')[1].strip().lower()
    algorithms[alg_name]=alg_path
    '''
    # or load all model
    if alg_name and alg_name.find('_') != -1:
        alg_type = alg_name.split('_')[0]
    else:
        alg_type = alg_name
    switcher = {'tf': load_tf_model, 'fasttext': load_fasttext_model}
    func = switcher.get(alg_type, lambda: "Invalid month")
    func(alg_name, alg_path)'''

def reload_model(algorithm):
    model=None
    if algorithm.startswith('tf') and algorithm in algorithms:
        model=load_tf_model(algorithm, algorithms[algorithm])
    elif algorithm.startswith('fasttext') and algorithm in algorithms:
        model=load_fasttext_model(algorithm, algorithms[algorithm])
    return model

# ----------------------------------------------------------------------------------------
# app
# ----------------------------------------------------------------------------------------

resp_info_algorithms = api.model('resp_info', {'algorithms': fields.List(fields.String())})
@ns.route("/algorithms")
class getAlgorithms(Resource):
    @api.marshal_with(resp_info_algorithms)
    def get(self):
        alg_response=[]
        for key, value in algorithms.items(): 
            alg_response.append(key)
        return {'algorithms':alg_response}, 200

# define request payload json structure
req_info = {}
req_info['sentence'] = fields.String(required = True, example = "This is a test")
req_info['algorithm'] = fields.String(required = True, example = "tf")

# define return payload json structure
req_info_model = api.model("req_info",req_info)
resp_info_model = api.model('resp_info', {'emb_vect': fields.List(fields.Float())})

@ns.route("/")
class convert2Embedding(Resource):
    @ns.expect(req_info_model)
    @api.marshal_with(resp_info_model)
    def post(self):
        global current_algorithm,model
        sentence = request.json['sentence']
        algorithm = request.json['algorithm']
        if sentence:
            sentence=sentence.replace('\n','').replace("\r","")
        if algorithm:
            algorithm = algorithm.lower()
        if current_algorithm==None or current_algorithm!=algorithm:
            model = reload_model(algorithm)
            current_algorithm = algorithm
        if algorithm.startswith('tf') and model and current_algorithm:
            result=model([sentence]).numpy()
            retV = {"emb_vect": result[0].tolist()}
        elif algorithm.startswith('fasttext') and model and current_algorithm:
            #algorithm=='fasttext_eng' or algorithm=='fasttext_ara'
            retV = {'emb_vect': model.get_sentence_vector(sentence)}
        return retV,200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
