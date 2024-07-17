FROM python:3.10

WORKDIR /code

ENV CHATBOT_NAME="Chatbot"
ENV LLM_MODEL_NAME="mistral:instruct"
ENV LLM_TEMPERATURE="0.1"
ENV VECTORSTORES_DIR="./vectorstores"
ENV EMBEDDINGS_MODEL_NAME="BAAI/bge-large-en-v1.5"
ENV NB_RETRIVED_DOCS="4"

COPY ./setup.py /code/setup.py
COPY ./pyproject.toml /code/pyproject.toml


COPY ./src /code/src
RUN pip install /code[ui]

COPY ./app /code/app
COPY ./.streamlit /code/.streamlit
COPY ./static /code/static
RUN mkdir ./vectorstores

CMD ["streamlit", "run", "app/app.py", "--server.port", "80"]