
import os

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()


class QueryRewriter:

    def __init__(self):

        # Load Groq API key
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Check your .env file."
            )

        # Load Groq model
        self.model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b"
        )

        # Initialize Groq client
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        print("Groq Query Rewriter initialized successfully!")

    def rewrite_query(
        self,
        query: str,
        conversation_context: str = ""
    ) -> str:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        # System instructions
        system_prompt = """
        You are an expert query rewriting assistant
        for a Retrieval-Augmented Generation system.

        Your task is to rewrite the user's question into
        one clear, standalone search query.

        Follow these rules:

        1. Preserve the original meaning and intent.
        2. Resolve ambiguous references using conversation context.
        3. Include relevant technical terminology when supported
           by the context.
        4. Do not answer the user's question.
        5. Do not invent facts.
        6. Avoid unrelated information.
        7. Return only the rewritten search query.
        """

        # User prompt
        user_prompt = f"""
        Conversation context:
        {conversation_context or "No additional context provided."}

        Original user question:
        {query}

        Rewrite this question for document retrieval.
        """

        # Call Groq API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_completion_tokens=1024,
            reasoning_effort="low",
            extra_body={
                "include_reasoning": False
            }
        )

        # Extract response
        choice = response.choices[0]

        rewritten_query = (
            choice.message.content or ""
        ).strip()

        # Validate response
        if not rewritten_query:

            raise RuntimeError(
                "Groq returned empty content. "
                f"Finish reason: {choice.finish_reason}. "
                f"Model: {self.model}"
            )

        # Remove unnecessary quotation marks
        rewritten_query = rewritten_query.strip('"').strip("'")

        return rewritten_query
