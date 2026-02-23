import os
from dotenv import load_dotenv  # Load environment variables from .env file


from langchain_openai import ChatOpenAI  # OpenAI chat model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # Message types


# Load environment variables from .env file
# This reads OPENAI_API_KEY without hardcoding it in code
load_dotenv()


# Initialize the chat model
# temperature=0.7 means moderate creativity (0=focused, 1=creative) Class Chat():
chat = ChatOpenAI(
   model="gpt-4o-mini",  # Fast, cheap, modern
   temperature=0.7,
   api_key=os.getenv("OPENAI_API_KEY"),
)


def simple_chat():
   """
   A basic chat function that maintains conversation history manually
   """
   
   # Manually maintained conversation history
   conversation_history = [
       SystemMessage(content="You are a helpful assistant.")
   ]


   print("Simple Chatbot (type 'quit' to exit)")
   print("-" * 50)


   while True:
       user_input = input("\nYou: ")


       if user_input.lower() == "quit":
           print("Goodbye 👋")
           break


       # Add user message
       conversation_history.append(HumanMessage(content=user_input))


       # Call the model with full history
       # LIMITATION: Entire history must be passed each time
       response = chat.invoke(conversation_history)


       # response is already an AIMessage
       conversation_history.append(response)


       print(f"\nBot: {response.content}")


       # LIMITATIONS:
       # - History grows without bounds
       # - No memory summarization
       # - No tools
       # - No multi-step reasoning / workflows




if __name__ == "__main__":
   simple_chat()