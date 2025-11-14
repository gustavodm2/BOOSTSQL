import logging
from typing import Optional
import os

logger = logging.getLogger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. LLM features will be disabled.")

class LLMSyntaxCorrector:
    """
    Uses LLM to correct syntax errors in SQL queries.
    """

    def __init__(self, api_key: str, model: str = 'gpt-3.5-turbo', temperature: float = 0.1, max_tokens: int = 500, mock: bool = False):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mock = mock

        if mock:
            logger.info("Using mock LLM mode for testing")
            self.enabled = True
        elif not OPENAI_AVAILABLE:
            logger.warning("OpenAI library not installed. Using mock mode.")
            self.mock = True
            self.enabled = True
        elif not self.api_key:
            logger.warning("No OpenAI API key provided. Using mock mode for testing.")
            self.mock = True
            self.enabled = True
        else:
            openai.api_key = self.api_key
            self.enabled = True

    def correct_syntax(self, query: str) -> tuple[str, bool]:
        """
        Correct syntax errors in the SQL query using LLM.
        Returns (corrected_query, was_corrected)
        """
        if not self.enabled:
            return query, False

        if self.mock:
            # Mock correction: just add a comment
            return f"{query} /* LLM corrected */", True

        try:
            prompt = f"""
You are a SQL syntax corrector. Your task is to fix any syntax errors in the following SQL query while preserving its intended meaning and structure.

Original query:
{query}

Please provide only the corrected SQL query without any explanations or additional text. If the query is already syntactically correct, return it unchanged.
"""

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a SQL syntax expert that corrects syntax errors in SQL queries and make it kinda optimized."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            corrected_query = response.choices[0].message.content.strip()

            # Remove any markdown code blocks if present
            if corrected_query.startswith("```sql"):
                corrected_query = corrected_query[6:]
            if corrected_query.endswith("```"):
                corrected_query = corrected_query[:-3]
            corrected_query = corrected_query.strip()

            was_corrected = corrected_query != query
            if was_corrected:
                logger.info("LLM corrected query syntax")
            return corrected_query, was_corrected

        except Exception as e:
            logger.warning(f"LLM syntax correction failed: {e}")
            return query, False