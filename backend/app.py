from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)

    # Register Blueprints
    from tasks.dog_classification.routes import dog_classification_bp
    from tasks.car_detection.routes import car_detection_bp
    
    app.register_blueprint(dog_classification_bp, url_prefix='/dog_classification')
    app.register_blueprint(car_detection_bp, url_prefix='/car_detection')
    
    return app

app = create_app()
CORS(app)

if __name__ == '__main__':
    app.run(debug=True)
