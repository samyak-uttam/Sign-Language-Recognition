from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
	return send_from_directory('./templates', 'index.html')

@app.route('/<path:path>', methods=['GET'])
def sen_root(path):
	return send_from_directory('./templates', path)

if __name__ == '__main__':
	app.run(debug=True)