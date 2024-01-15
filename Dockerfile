FROM python:3.10

WORKDIR /code

COPY ./setup.py /code/setup.py
COPY ./pyproject.toml /code/pyproject.toml


COPY ./src /code/src
RUN pip install /code[ui]

COPY ./app /code/app
COPY ./static /code/static
RUN mkdir ./vectorstores

CMD ["streamlit", "run", "app/app.py", "--server.port", "8080"]