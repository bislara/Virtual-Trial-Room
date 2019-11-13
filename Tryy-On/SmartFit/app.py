from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/api/infer', methods=['POST'])
def infer():
    if not all(i in request.files for i in ['person', 'clothing']):
        return jsonify(error = 'Incorrect payload!')

    if not (request.files['person'].mimetype[:5] == request.files['clothing'].mimetype[:5] == "image"):    
        return jsonify(error = 'Inputs must be images!')

    person_img = request.files['person']
    clothing_img = request.files['clothing']

    person_img.save('./inputs/input_person.jpg')
    clothing_img.save('./inputs/input_clothing.jpg')

    os.system('./run_smartfit.sh ./inputs/input_person.jpg ./inputs/input_clothing.jpg')

    if not os.path.isfile('output/output.png'):
        return jsonify(error = '500: Internal server error')

    return send_from_directory('./output/', 'output.png')


if __name__ == '__main__':
    app.run(debug=True)
