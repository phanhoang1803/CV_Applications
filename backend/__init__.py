from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Register Blueprints
    from .tasks.car_detection.routes import car_detection_bp
    from .tasks.dog_classification.routes import dog_classification_bp
    
    app.register_blueprint(car_detection_bp)
    app.register_blueprint(dog_classification_bp)
    
    return app
