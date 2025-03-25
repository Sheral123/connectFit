from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from app.services.ai_utils import GeminiLLM
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

def generate_personalized_plan(preferences: dict) -> str:
    # Extensive JSON-like structure for demonstration.
    docs_json = [
        {
            "topic": "Meal Plans",
            "content": (
                "A balanced meal plan includes lean proteins such as chicken, fish, legumes, and tofu; "
                "a wide variety of fresh vegetables like broccoli, spinach, and carrots; "
                "whole grains including brown rice, quinoa, and oats; and healthy fats from avocado, nuts, seeds, and olive oil. "
                "The plan should incorporate seasonal produce, minimize processed foods, and offer diverse recipes to cover all essential nutrients."
            )
        },
        {
            "topic": "Workout Plans",
            "content": (
                "A comprehensive workout plan should combine cardio (running, cycling, HIIT) with strength training (both compound and isolation exercises) "
                "and flexibility exercises such as yoga or dynamic stretching. "
                "Structured rest days and progressive overload are key to preventing plateaus and ensuring continued improvement."
            )
        },
        {
            "topic": "Weight Loss Strategies",
            "content": (
                "For effective weight loss, it's important to achieve a sustainable calorie deficit by monitoring portion sizes and focusing on nutrient-dense foods. "
                "Combining dietary adjustments with regular physical activity—whether through structured exercise or lifestyle changes—can significantly enhance results. "
                "Techniques like intermittent fasting or time-restricted eating may also support a healthy weight loss process."
            )
        },
        {
            "topic": "Nutritional Variety and Balance",
            "content": (
                "Ensuring nutritional variety means consuming a diverse range of foods to cover all vitamins, minerals, and macronutrients. "
                "This includes rotating different protein sources, eating a spectrum of colorful fruits and vegetables, and balancing carbohydrates and fats to maintain energy levels and promote overall health."
            )
        }
    ]
    
    # Convert the JSON-like structure into a list of text documents for indexing.
    docs = [f"Topic: {doc['topic']}\nContent: {doc['content']}" for doc in docs_json]
    
    # Initialize the embeddings using FastEmbedEmbeddings.
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # Create a FAISS vector store from the list of text documents.
    vector_store = FAISS.from_texts(docs, embeddings)
    
    # Create a retriever from the vector store.
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # Retrieve documents relevant to the preferences.
    # (Here, for simplicity, we convert the preferences dict to a string.)
    retrieved_docs = retriever.get_relevant_documents(str(preferences))
    retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Define a prompt that includes the user preferences and the retrieved context.
    prompt_template = PromptTemplate(
        input_variables=["preferences", "retrieved_context"],
        template=(
            "User Preferences: {preferences}\n\n"
            "Retrieved Context: {retrieved_context}\n\n"
            "Based on the above, generate a detailed weekly personalized meal and workout plan."
        )
    )
    
    # Instantiate the Gemini LLM and set up the chain.
    llm = GeminiLLM()
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(preferences=str(preferences), retrieved_context=retrieved_context)
