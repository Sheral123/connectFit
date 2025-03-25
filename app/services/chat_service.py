from langchain import LLMChain, PromptTemplate
from app.services.ai_utils import GeminiLLM

def get_chat_response(message: str) -> str:
    llm = GeminiLLM()
    prompt_template = PromptTemplate(
        input_variables=["message"],
        template="You are a friendly and knowledgeable fitness and wellness assistant for ConnectFit.\nUser: {message}\nAssistant:"
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(message=message)
