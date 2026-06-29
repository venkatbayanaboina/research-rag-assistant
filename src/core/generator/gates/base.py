class GeminiQuotaExhaustedError(Exception):
    """Raised when the primary Gemini API has completely exhausted its daily request quota."""
    pass

class BaseModelGate:
    def generate(self, prompt, system_instruction=None, response_mime_type=None, temperature=0.3):
        """Generates text from prompts. Must be overridden by subclasses."""
        raise NotImplementedError()
