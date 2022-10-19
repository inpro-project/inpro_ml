import json
import requests


def send_post_api():
    url = "http://localhost:8000/predict/image"

    try:
        response = requests.post(
            url,
            files = {'file': open('../image/animal.jpg', 'rb')}
            )
        print(response.json())

    except Exception as e:
        print(e)