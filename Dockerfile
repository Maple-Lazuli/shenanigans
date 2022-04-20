FROM tensorflow/tensorflow:2.8.0-gpu

RUN pip install --upgrade pip

COPY *.py /opt/app/

COPY requirements.txt /opt/app

RUN mkdir /opt/app/model

RUN mkdir /opt/app/reports

RUN mkdir /opt/app/reports/images

RUN mkdir /opt/app/raw_data

RUN mkdir /opt/app/tf_records

RUN pip install -r /opt/app/requirements.txt

RUN chmod -R 777 /opt/app

ENTRYPOINT /bin/bash