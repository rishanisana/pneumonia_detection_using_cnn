from flask import Flask , redirect , url_for , request , render_template, session
from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

model_file = "model.h5"
model = load_model(model_file)

app = Flask(__name__,template_folder='templates')
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 


CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, username, password):
        self.username = username
        self.password = password
        
@app.route('/', methods=['GET','POST'])
@app.route('/index')
def index():
    if session.get('logged_in'):
        return redirect('home')
    else:
        return render_template('index.html')

@app.route('/register/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            db.session.add(User(username=request.form['username'], password=request.form['password']))
            db.session.commit()
            return redirect(url_for('login'))
        except:
            return render_template('register.html', message="User Already Exists")
    else:
        return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        u = request.form['username']
        p = request.form['password']
        data = User.query.filter_by(username=u, password=p).first()
        if data is not None:
            session['logged_in'] = True
            return redirect('home')
        return render_template('login.html', message="Incorrect Details")

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session['logged_in'] = False
    return redirect(url_for('index'))


def makePredictions(path):
    '''
    Method to predict the probability of pneumonia in the uploaded image
    '''
    img = Image.open(path)
    img_d = img.resize((224,224))  # Resize for model
    
    # Convert grayscale to RGB
    if len(np.array(img_d).shape) < 3:
        rgbimg = Image.new("RGB", img_d.size)
        rgbimg.paste(img_d)
    else:
        rgbimg = img_d
    
    rgbimg = np.array(rgbimg, dtype=np.float64) / 255.0  # Normalize
    rgbimg = rgbimg.reshape((1, 224, 224, 3))  # Reshape for model
    
    # Make prediction
    predictions = model.predict(rgbimg)[0]
    
    # Extract probabilities
    pneumonia_prob = float(predictions[1]) * 100  # Convert to percentage
    health_prob = float(predictions[0]) * 100
    
    # Conditions for classification
    if pneumonia_prob < 40:
        result = f"✅ Healthy (Low Risk - {health_prob:.2f}%)"
    elif 40 <= pneumonia_prob <= 70:
        result = f"⚠️ Mild Chances of Pneumonia ({pneumonia_prob:.2f}%)"
    else:
        result = f"❌ High Chances of Pneumonia ({pneumonia_prob:.2f}%)"

    return result


@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'img' not in request.files:
            return render_template('home.html', filename="3631348.jpg", message="Please upload a file")
        
        f = request.files['img']
        filename = secure_filename(f.filename)

        if filename == '':
            return render_template('home.html', filename="3631348.jpg", message="No file selected")
        if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg') or filename.lower().endswith('.png')):
            return render_template('home.html', filename="3631348.jpg", message="Please upload a .png, .jpg, or .jpeg file")
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)

        predictions = makePredictions(file_path)
        
        return render_template('home.html', filename=filename, message=predictions, show=True)
    
    return render_template('home.html', filename='3631348.jpg')



if __name__ == "__main__":
    app.secret_key = "ThisIsNotASecret:p"
    with app.app_context():
        db.create_all()
        app.run(debug=True)
        