FROM tensorflow/tensorflow:nightly-devel-gpu-py3
RUN mkdir /src
WORKDIR /src
COPY . /src

RUN pip install scikit-image pillow scipy flask flask-cors

CMD ["python", "api.py"]
