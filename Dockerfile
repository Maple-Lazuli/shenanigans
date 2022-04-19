FROM tensorflow/tensorflow:2.8.0-gpu

RUN pip install --upgrade pip

COPY . /opt/app/

RUN pip install -r /opt/app/requirements.txt

ENTRYPOINT /bin/bash