FROM continuumio/anaconda3
ENV PYTHONUNBUFFERED=1
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
COPY conda-req.txt /code/
RUN pip install -r requirements.txt --ignore-installed

# RUN conda install --file conda-req.txt
RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

RUN mkdir model && cd model && wget https://pjreddie.com/media/files/yolov3.weights && cd ..


RUN apt update -y
RUN apt install libgl1-mesa-glx -y
RUN apt install libturbojpeg -y

# Ubuntu renamed the libturbojpeg package

COPY . /code/

CMD [ "python", "main.py", "--no-gpus" ]