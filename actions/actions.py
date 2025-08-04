import openai
import os
from dotenv import load_dotenv
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

load_dotenv()  # Load OPENAI_API_KEY from .env file
api_key = os.getenv("OPENAI_API_KEY")

class ActionLLMFallback(Action):
    def name(self) -> Text:
        return "action_llm_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        openai.api_key = os.getenv("OPENAI_API_KEY")
        query = tracker.latest_message.get('text')

        resume_context = """
        Ira Pathak is a Computer Science student at Georgia Tech with a FinTech minor. 
        She has experience at AllzGo and Bhaifi, working on backend and network tools.
        Projects include: EnGarde (Django + ML), PrintsNCuts (MERN), FinBERT sentiment analysis.
        Skills: Python, React, LangChain, ML, TensorFlow, RAG, Selenium, SQL, MongoDB.
        Interests: AI, Finance, iOS Dev, camping.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers based on the resume below.\n" + resume_context},
                    {"role": "user", "content": query}
                ]
            )
            answer = response['choices'][0]['message']['content']
        except Exception as e:
            answer = f"LLM error: {str(e)}"

        dispatcher.utter_message(text=answer)
        return []
