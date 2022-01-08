FROM python:3.8-buster

COPY api.py api.py
COPY on_the_list_crm/onthelist_segmentation.py on_the_list_crm/onthelist_segmentation.py
COPY on_the_list_crm/product_cat_and_gender.py on_the_list_crm/product_cat_and_gender.py 
COPY requirements.txt requirements.txt

RUN pip install -U pip
RUN pip install -r requirements.txt

CMD uvicorn api:app --host 0.0.0.0 --port $PORT
