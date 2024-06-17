from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

encoder_model1 = 'avsolatorio/GIST-small-Embedding-v0'

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["disease_name"] = record.get("disease_name")
    metadata["content"] = f"""The disease name is {record.get('disease_name')}. {record.get('about')}. 
        The main causes for it are {record.get('causes')}. The symptoms include {record.get('symptoms')}. 
        Available treatments are {record.get('treatments')}. For more information, visit {record.get('source_link')}."""
    return metadata

def get_database(path, encoder_model):
    loader = JSONLoader(
        file_path=path,
        jq_schema='.[]',
        content_key="symptoms",
        metadata_func=metadata_func
    )

    data = loader.load()
    db = FAISS.from_documents(data, encoder_model)
    return db

def get_encoder_model(model_name = 'avsolatorio/GIST-small-Embedding-v0'):
    encoder_model = HuggingFaceEmbeddings(model_name=model_name)
    return encoder_model

def vector_for_text_file(path, encoder_model):
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n",],
        chunk_size=1024,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    with open(path) as f:
        data = f.read()
    
    texts = text_splitter.create_documents([data])

    db = FAISS.from_documents(texts, encoder_model)
    return db

