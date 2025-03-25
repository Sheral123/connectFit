from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from app.services.ai_utils import GeminiLLM

# Define a custom prompt template that incorporates conversation history.
# The template expects two variables:
#   - chat_history: the conversation so far.
#   - input: the new user message.
custom_prompt_template = PromptTemplate(
    input_variables=["chat_history", "input"],
    template=(
        "You are a friendly and knowledgeable fitness and wellness assistant for ConnectFit.\n\n"
        "Conversation History:\n{chat_history}\n\n"
        "User: {input}\n"
        "Assistant:"
    )
)

# Create a memory buffer to store conversation history.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def get_chat_response(message: str) -> str:
    # Instantiate your custom LLM (GeminiLLM).
    llm = GeminiLLM()
    
    # Create a ConversationChain with the custom prompt template and memory.
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=custom_prompt_template,
        verbose=True
    )
    
    # The chain automatically combines previous conversation with the new message.
    response = conversation.predict(input=message)
    return response
