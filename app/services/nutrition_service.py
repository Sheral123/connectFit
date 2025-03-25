from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from app.services.ai_utils import GeminiLLM

# Define a custom prompt template for nutritional advice.
# This template includes conversation history and the new query.
nutrition_prompt_template = PromptTemplate(
    input_variables=["chat_history", "input"],
    template=(
        "You are a nutrition expert providing clear, concise, and evidence-based nutritional advice.\n\n"
        "Conversation History:\n{chat_history}\n\n"
        "User Query: {input}\n"
        "Nutrition Expert:"
    )
)

# Create a memory buffer to store the conversation history.
nutrition_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def get_nutrition_advice(query: str) -> str:
    # Instantiate your GeminiLLM.
    llm = GeminiLLM()
    
    # Create a ConversationChain with the custom nutrition prompt and memory.
    nutrition_conversation = ConversationChain(
        llm=llm,
        memory=nutrition_memory,
        prompt=nutrition_prompt_template,
        verbose=True
    )
    
    # Predict the response by passing in the new query.
    response = nutrition_conversation.predict(input=query)
    return response
