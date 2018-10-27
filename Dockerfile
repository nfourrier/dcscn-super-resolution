FROM tensorflow/tensorflow:nightly-devel-gpu-py3
RUN mkdir /src
WORKDIR /src
COPY . /src

RUN pip install scikit-image pillow scipy>=1.1.0 flask flask-cors

CMD ["python", "api.py"]
