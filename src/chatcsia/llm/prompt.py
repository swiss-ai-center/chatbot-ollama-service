from langchain.prompts import PromptTemplate


def prepare_prompt(language: str = "en") -> PromptTemplate:
    if language == "fr":
        return prepare_prompt_fr()
    elif language == "en":
        return prepare_prompt_en()
    elif language == "de":
        return prepare_prompt_de()
    elif language == "it":
        return prepare_prompt_it()

    raise ValueError(f"Language {language} not supported")


def prepare_prompt_en() -> PromptTemplate:
    template = """<s>[INST]<<SYS>>Use the following pieces of context to answer the question at the end.
    Don't try to make up an answer and only use the information you know.
    If images are provided, use them to also answer the question.
    Use three sentences maximum and keep the answer as concise as possible.
    You must answer in english.{context}<</SYS>>
    {question}[/INST]"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )


def prepare_prompt_fr() -> PromptTemplate:
    template = """<s>[INST]<<SYS>>Utilise les éléments de contexte suivants pour répondre à la question finale.
    N'essayez pas d'inventer une réponse et n'utilise que les informations que tu connais.
    Si des images sont fournies, utilisez-les pour répondre à la question.
    Utilise trois phrases au maximum et soit aussi concis que possible dans ta réponse.
    Tu dois répondre en français.{context}<</SYS>>
    {question}[/INST]"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )


def prepare_prompt_de() -> PromptTemplate:
    template = """<s>[INST]<<SYS>>Beantworten Sie die Frage am Ende des Textes anhand
    der folgenden Kontextinformationen.
    Versuchen Sie nicht, sich eine Antwort auszudenken und verwenden Sie nur die Informationen, die Sie kennen.
    Wenn Bilder zur Verfügung gestellt werden, verwenden Sie diese ebenfalls, um die Frage zu beantworten.
    Verwenden Sie maximal drei Sätze und fassen fassen Sie die Antwort so kurz wie möglich zusammen.
    Sie müssen auf Deutsch antworten.{context}<</SYS>>
    {question}[/INST]"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )


def prepare_prompt_it() -> PromptTemplate:
    template = """<s>[INST]<<SYS>>Utilizza i seguenti elementi di contesto per rispondere alla domanda finale.
    Non cercare di inventare una risposta e utilizza solo le informazioni che conosci.
    Se vengono fornite delle immagini,usale per rispondere alla domanda.
    Utilizza al massimo tre frasi e dai una risposta la più concisa possibile.
    È necessario rispondere in italiano.{context}<</SYS>>
    {question}[/INST]"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
