from flask import jsonify, Flask, send_file, request, make_response
from werkzeug.utils import secure_filename
import os
from PIL import Image
import gzip
import numpy as np
from flask_cors import CORS
import time


import cv2
import gc
import json
import subprocess, sys


from al import train, recommend_superpixels


app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = "static/files"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(APP_ROOT, app.config["UPLOAD_FOLDER"])




@app.route('/stl', methods=['POST'])
def stl():
    TEST_REGION = int(request.args.get('testRegion', 1))
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        subprocess.check_output(['./hmm', f.filename, f'./stl/Region_{TEST_REGION}.stl', '-z', '500', '-t', '10000000'])
        print("testRegion: ", TEST_REGION)
        payload = make_response(send_file(f'./stl/Region_{TEST_REGION}.stl'))
        payload.headers.add('Access-Control-Allow-Origin', '*')
        # os.remove('a.stl')
        # os.remove(f.filename)

        return payload
    
@app.route('/superpixel', methods=['GET'])
def superpixel():
    recommend = request.args.get('recommend')
    initial = int(request.args.get('initial', 0))

    entropy = int(request.args.get('entropy', 0))
    probability = int(request.args.get('probability', 0))
    cod = int(request.args.get('cod', 0))

    transformation_agg = request.args.get('transformation_agg', '').strip()
    superpixel_agg = request.args.get('superpixel_agg', '').strip()

    student_id = request.args.get('taskId', '').strip()
    TEST_REGION = int(request.args.get('testRegion', 1))
    use_forest = int(request.args.get('use_forest', 1))
    
    print("superpixel_agg, transformation_agg: ")
    print(superpixel_agg, transformation_agg)
    print(student_id)
    
    print("entropy, probability, cod: ")
    print(entropy, probability, cod)
    
    print("use_forest: ", use_forest)

    # read AL cycle from txt file
    try:
        with open(f"./users/{student_id}/al_cycles/R{TEST_REGION}.txt", 'r') as file:
            content = file.read()
            al_cycle = int(content) 
    except FileNotFoundError:
        al_cycle = 0

    # to handle the situation where a user does AL for a while, terminated and starts again (all the process has to be completed in 1 go, there can be no break in between)
    if initial:
        try:
            with open(f"./users/{student_id}/resume_epoch/R{TEST_REGION}.txt", 'w') as file:
                file.write(str(0))
        except FileNotFoundError:
            pass

        try:
            with open(f"./users/{student_id}/al_cycles/R{TEST_REGION}.txt", 'w') as file:
                file.write(str(0))
                al_cycle = 0
        except FileNotFoundError:
            pass

        try:
            with open(f"./users/{student_id}/al_iters/R{TEST_REGION}.txt", 'w') as file:
                file.write(str(0))
        except FileNotFoundError:
            pass

    metrices = {}
    if int(recommend):
        start_time = time.time()
        
        metrices, metrices_unlabeled, _, _ = recommend_superpixels(TEST_REGION, entropy, probability, cod, transformation_agg, superpixel_agg, student_id, al_cycle, use_forest=use_forest)

        end_time = time.time()
        elapsed_time = (end_time - start_time)/60
        elapsed_time = float("{:.2f}".format(elapsed_time))

        metrices += "\n"
        metrices += f"Elapsed Time: {elapsed_time} minutes"

        metrices += "\n\n"
        metrices += metrices_unlabeled

        file_path = f"./users/{student_id}/output/Region_{TEST_REGION}_Metrics_C{al_cycle}.txt"
        with open(file_path, "w") as fp:
            fp.write(metrices)

    payload = make_response(send_file(f'./users/{student_id}/output/R{TEST_REGION}_superpixels_test.png'))
    payload.headers.add('Access-Control-Allow-Origin', '*')

    return payload


@app.route('/pred')
def pred():
    TEST_REGION = int(request.args.get('testRegion', 1))
    student_id = request.args.get('taskId')

    payload = make_response(send_file(f'./users/{student_id}/output/R{TEST_REGION}_pred_test.png'))
    payload.headers.add('Access-Control-Allow-Origin', '*')
    return payload


@app.route('/confidence')
def confidence():
    TEST_REGION = int(request.args.get('testRegion', 1))

    payload = make_response(send_file(f'./data_al/forest/R{TEST_REGION}_forest.png'))
    # payload.json = json_data
    payload.headers.add('Access-Control-Allow-Origin', '*')

    return payload

@app.route('/forest-json', methods=['GET'])
def forest_json():
    TEST_REGION = int(request.args.get('testRegion', 1))
    file_path = f"./data_al/forest/Region_{TEST_REGION}_forest_json.json"

    forests = {}
    with open(file_path, 'r') as json_file:
        forests = json.load(json_file)

    payload = make_response(jsonify(forests), 200)
    payload.headers.add('Access-Control-Allow-Origin', '*')

    return payload

@app.route('/gt-json', methods=['GET'])
def gt_json():
    TEST_REGION = int(request.args.get('testRegion', 1))
    file_path = f"./data_al/gt/Region_{TEST_REGION}_gt_json.json"

    gts = {}
    with open(file_path, 'r') as json_file:
        gts = json.load(json_file)

    payload = make_response(jsonify(gts), 200)
    payload.headers.add('Access-Control-Allow-Origin', '*')

    return payload

@app.route('/metrics-json', methods=['GET'])
def metrics_json():
    TEST_REGION = int(request.args.get('testRegion', 1))
    student_id = request.args.get('taskId').strip()

    print("student_id: ", student_id)

    try:
        with open(f"./users/{student_id}/al_cycles/R{TEST_REGION}.txt", 'r') as file:
            content = file.read()
            al_cycle = int(content) - 1
    except FileNotFoundError:
        al_cycle = 0

    if al_cycle < 0:
        al_cycle = 0

    file_path = f"./users/{student_id}/output/Region_{TEST_REGION}_Metrics_C{al_cycle}.txt"

    metrices = {}
    with open(file_path, 'r') as json_file:
        # metrices = json.load(json_file)
        lines = json_file.readlines()
        # metrices = " ".join(lines[:-16])
        metrices = " ".join(lines[:-14])

    payload = make_response(jsonify(metrices), 200)
    payload.headers.add('Access-Control-Allow-Origin', '*')

    return payload


@app.route('/retrain', methods=['POST', 'GET'])
def retrain():
    start_time = time.time()
    TEST_REGION = int(request.args.get('testRegion', 1))
    student_id = request.args.get('taskId').strip()
    file = request.files.get('image')

    entropy = int(request.args.get('entropy', 0))
    probability = int(request.args.get('probability', 0))
    cod = int(request.args.get('cod', 0))

    transformation_agg = request.args.get('transformation_agg', 'avg').strip()
    superpixel_agg = request.args.get('superpixel_agg', 'avg').strip()

    use_sc_loss = int(request.args.get('sc_loss', 1))
    use_cod_loss = int(request.args.get('cod_loss', 1))
    use_forest = int(request.args.get('use_forest', 1))
    
    print("entropy, probability, cod: ")
    print(entropy, probability, cod)
    
    print("sc_loss, cod_loss, use_forest: ")
    print(use_sc_loss, use_cod_loss, use_forest)

    # read cycle from txt file
    try:
        with open(f"./users/{student_id}/al_cycles/R{TEST_REGION}.txt", 'r') as file:
            content = file.read()
            al_cycle = int(content) 
    except FileNotFoundError:
        al_cycle = 0

    # read iters from txt file
    try:
        with open(f"./users/{student_id}/al_iters/R{TEST_REGION}.txt", 'r') as file:
            content = file.read()
            al_iters = int(content) 
    except FileNotFoundError:
        al_iters = 0

    if file:
        print('image is here')
        file = request.files['image']

        # Process the file as needed, for example, save it to the server
        file.save(f'./users/{student_id}/output/R{TEST_REGION}_labels.png')

        train(TEST_REGION, entropy, probability, cod, transformation_agg, superpixel_agg, student_id, al_cycle, al_iters, use_sc_loss, use_cod_loss, use_forest)

        payload = make_response(jsonify({'status': 'success', 'taskId': student_id}), 200)
        payload.headers.add('Access-Control-Allow-Origin', '*')

        with open(f"./status_{student_id}.txt", 'w') as file:
            file.write("completed")

        return payload

    payload = make_response(jsonify({'status': 'error', 'taskId': student_id}), 400)
    payload.headers.add('Access-Control-Allow-Origin', '*')

    with open(f"./status_{student_id}.txt", 'w') as file:
        file.write("error")

    return payload


@app.route('/check-status', methods=['GET'])
def check_status():
    student_id = request.args.get('taskId').strip()
    TEST_REGION = int(request.args.get('testRegion', 1))

    print("student_id: ", student_id)

    # logic to check the status of the task
    with open(f"./users/{student_id}/R{TEST_REGION}_status.txt", 'r') as file:
        status = file.read()
    
    print("status: ", status)

    payload = make_response(jsonify({'status': status}), 200)
    payload.headers.add('Access-Control-Allow-Origin', '*')

    return payload

    
if __name__ == '__main__':
   app.run()