import uvicorn
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import os
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('../model/mobilenetv2_imagenet_inpro')
app = FastAPI()

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    contents = await file.read()
    image_dir = os.path.join("../image", file.filename)
    with open(image_dir, "wb") as fp:
        fp.write(contents)
    cur_image = tf.keras.preprocessing.image.load_img(
        image_dir, target_size=(224, 224))
    cur_image = tf.keras.preprocessing.image.img_to_array(cur_image)
    x = np.expand_dims(cur_image, axis=0)
    prediction = model.predict(x)

    max_v = 0
    max_i = 0

    for i, v in enumerate(prediction[0]):
        if v > max_v:
            max_v = v
            max_i = i

    if max_i < 2:
        pred = {
            "is_valid": False
        }
    else:
        pred = {
            "is_valid": True
        }
    os.remove(image_dir)
    return pred

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
