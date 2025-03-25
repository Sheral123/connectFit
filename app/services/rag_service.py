from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from app.services.ai_utils import GeminiLLM
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

def generate_personalized_plan(preferences: dict) -> str:
    # Sample documents for demonstration.
    docs = [
        "A balanced meal plan should include lean proteins, fresh vegetables, whole grains, and healthy fats.",
        "A comprehensive workout plan should incorporate cardio, strength training, and flexibility exercises.",
        "For weight loss, combining a calorie deficit with regular physical activity is essential.",
        "Variety in meals ensures all essential nutrients are consumed."
    ]
    
    # Initialize the embeddings (using a sentence-transformers model)
     

    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # Create a FAISS vector store from the documents.
    vector_store = FAISS.from_texts(docs, embeddings)
    
    # Create a retriever from the vector store.
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # Retrieve documents relevant to the preferences.
    # (For simplicity, we convert preferences to string. In practice, you may use a specific query.)
    retrieved_docs = retriever.get_relevant_documents(str(preferences))
    retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Define a prompt that incorporates user preferences and retrieved context.
    prompt_template = PromptTemplate(
        input_variables=["preferences", "retrieved_context"],
        template=(
            "User Preferences: {preferences}\n\n"
            "Retrieved Context: {retrieved_context}\n\n"
            "Based on the above, generate a detailed weekly personalized meal and workout plan."
        )
    )
    
    llm = GeminiLLM()
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(preferences=str(preferences), retrieved_context=retrieved_context)
