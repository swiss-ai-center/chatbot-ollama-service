from langchain.prompts import PromptTemplate


def prepare_prompt() -> PromptTemplate:
    template = """<s>[INST]<<SYS>>Use the following pieces of context to answer the question at the end.
    Don't try to make up an answer and only use the information you know.
    If images are provided, use them to also answer the question.
    Use three sentences maximum and keep the answer as concise as possible.
    You must answer in English.{context}<</SYS>>
    {question}[/INST]"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
