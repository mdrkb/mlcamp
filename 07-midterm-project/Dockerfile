FROM python:3.9-alpine

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "./"]

EXPOSE 8080

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:8080", "predict:app"]