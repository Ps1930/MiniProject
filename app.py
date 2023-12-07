from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load your pre-trained model
CNN = tf.keras.models.load_model("CNN.h5", compile=False)

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']  # Get the uploaded image file
        if image_file:
            image_path = "./images/" + image_file.filename
            image_file.save(image_path)

            img = Image.open(image_path)
            img = img.resize((224, 224))
            
            # Ensure the image has three color channels (convert to RGB if needed)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            iArray = img_to_array(img)
            iArray = np.expand_dims(iArray, axis=0)

            p = CNN.predict(iArray)
            score = tf.nn.softmax(p[0])
            cl_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
            predicted_class = cl_labels[np.argmax(p)]
            
            # Convert the NumPy array to a list before passing it to the template
            scores_list = score.numpy().tolist()

            return render_template('index.html', prediction=predicted_class, scores=scores_list)
    
    return render_template('index.html', prediction=None, scores=None)

if __name__ == '__main__':
    app.run(debug=True)
