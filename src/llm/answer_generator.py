
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class AnswerGenerator:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is missing.")

        self.model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b"
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        print("Groq Answer Generator initialized successfully!")

    def generate_answer(
        self,
        question: str,
        contexts: list[dict]
    ) -> str:

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not contexts:
            return (
                "I could not find sufficient information "
                "in the available documents to answer this question."
            )

        # Prepare retrieved context
        context_parts = []

        for index, item in enumerate(contexts, start=1):

            metadata = item.get("metadata", {})

            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "Unknown")
            content = item.get("content", "")

            context_parts.append(
                f"""
                Context {index}
                Source: {source}
                Page: {page}

                Content:
                {content}
                """
            )

        retrieved_context = "\n\n".join(context_parts)

        system_prompt = """
        You are a Senior AI Engineer and technical interviewer.

        Your task is to answer technical interview questions
        using the retrieved document context.

        Follow these rules:

        1. Use the retrieved context as your primary evidence.
        2. Give clear, technically accurate explanations.
        3. Structure answers with headings or bullet points
           when appropriate.
        4. Include practical examples when useful.
        5. Do not invent project-specific facts.
        6. If the context does not contain enough information,
           clearly mention the limitation.
        7. Distinguish document facts from general technical
           explanations.
        8. Keep the answer relevant to the question.
        9. Do not claim that an unsupported fact came from
           the retrieved documents.
        """

        user_prompt = f"""
        Interview Question:
        {question}

        Retrieved Document Context:
        {retrieved_context}

        Generate a clear and detailed technical interview answer.
        """

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
            max_completion_tokens=2048,
            reasoning_effort="low",
            extra_body={
                "include_reasoning": False
            }
        )

        answer = (
            response.choices[0].message.content or ""
        ).strip()

        if not answer:
            raise RuntimeError(
                "Groq returned an empty answer. "
                f"Finish reason: {response.choices[0].finish_reason}"
            )

        return answer
