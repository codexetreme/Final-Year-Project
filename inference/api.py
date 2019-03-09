from flask import Flask
app = Flask(__name__)


@app.route('/')
def get_document():
    
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()