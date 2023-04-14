FROM python:3.8

RUN python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install torch==1.13.0+cpu torchvision==0.14.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install timm flask 
# RUN mkdir /flask-flower102-distinguish


# COPY app.py /flask-flower102-distinguish/app.py
# COPY model_best.pth.tar /flask-flower102-distinguish/model_best.pth.tar
# COPY Oxford-102_Flower_dataset_labels.txt /flask-flower102-distinguish/Oxford-102_Flower_dataset_labels.txt 

COPY . .

# RUN cd /flask-flower102-distinguish && flask run --host=0.0.0.0

CMD ["flask", "run", "--host=0.0.0.0"]


