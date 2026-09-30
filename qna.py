import requests

from config import GEMINI_API_KEY, GEMINI_MODEL, is_configured, logger


def answer_question_with_gemini(question: str) -> str:
    if not is_configured():
        return "Gemini API key is not configured. Add your GEMINI_API_KEY in the .env file."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": question}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500},
    }

    try:
        response = requests.post(url, headers={"x-goog-api-key": GEMINI_API_KEY}, json=payload, timeout=60)
        if response.status_code in {404, 429, 500, 502, 503, 504} and GEMINI_MODEL != "gemini-3.5-flash-lite":
            fallback_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"
            logger.warning(
                "Gemini model %s returned HTTP %d; retrying Q&A with gemini-3.5-flash-lite",
                GEMINI_MODEL,
                response.status_code,
            )
            response = requests.post(
                fallback_url,
                headers={"x-goog-api-key": GEMINI_API_KEY},
                json=payload,
                timeout=60,
            )
        response.raise_for_status()
        data = response.json()
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
        if text:
            return text.strip()
        return "I couldn't generate an answer from Gemini right now."
    except Exception as exc:
        logger.exception("Gemini Q&A request failed")
        return f"Error generating answer: {exc}"
