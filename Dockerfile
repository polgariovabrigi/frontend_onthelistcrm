FROM python:3.8-buster

COPY api.py api.py


COPY on_the_list_crm/onthelist_segmentation.py on_the_list_crm/onthelist_segmentation.py
COPY on_the_list_crm/product_cat_and_gender.py on_the_list_crm/product_cat_and_gender.py
COPY on_the_list_crm/vendor_cat.py on_the_list_crm/vendor_cat.py
COPY on_the_list_crm/cleansing_dataset.py on_the_list_crm/cleansing_dataset.py

COPY on_the_list_crm/kmean_model_05_01_2022_19h19.sav on_the_list_crm/kmean_model_05_01_2022_19h19.sav
COPY on_the_list_crm/nlp_model_05_01_2022_16h16.sav on_the_list_crm/nlp_model_05_01_2022_16h16.sav
COPY on_the_list_crm/product_cat_dict_05_01_2022_15h43.txt on_the_list_crm/product_cat_dict_05_01_2022_15h43.txt
COPY on_the_list_crm/tokenizer_for_nlp_model_05_01_2022_15h43.sav on_the_list_crm/tokenizer_for_nlp_model_05_01_2022_15h43.sav
COPY on_the_list_crm/vendor_cat_original_list.txt on_the_list_crm/vendor_cat_original_list.txt

COPY requirements.txt requirements.txt

RUN pip install -U pip
RUN pip install -r requirements.txt

CMD uvicorn api:app --host 0.0.0.0 --port $PORT
