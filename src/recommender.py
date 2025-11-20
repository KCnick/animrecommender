from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from src.prompt_template import get_anime_prompt

class AnimeRecommender:
    def __init__(self, retriever, api_key: str, model_name: str):
        self.llm = ChatGroq(api_key=api_key, model=model_name, temperature=0)
        self.prompt = get_anime_prompt()
        
        # Create the document chain that combines documents with the prompt
        document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        
        # Create the retrieval chain
        self.qa_chain = create_retrieval_chain(retriever, document_chain)

    def get_recommendation(self, query: str):
        # The new chain expects 'input' instead of 'query'
        result = self.qa_chain.invoke({"input": query})
        
        # Return the answer (and optionally source documents)
        return result['answer']