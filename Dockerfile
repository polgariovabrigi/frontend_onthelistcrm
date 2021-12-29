# FROM python:3.8-buster
FROM python:3.7-slim


COPY api.py api.py

RUN pip install -U pip
RUN pip install fastapi uvicorn

CMD uvicorn api:app --host 0.0.0.0 --port $PORT

# FROM python:3.7-slim
# ENV APP_HOME /app
# WORKDIR $APP_HOME
# COPY . ./
# RUN pip install pipenv
# RUN pipenv install --deploy --system
# CMD exec gunicorn --bind :$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker  --threads 8 app.main:app
