import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

def get_gemini_client():
    """Backward compatibility hook to fetch the active Google GenAI client."""
    from src.core.generator.gates import get_llm_gate
    from src.core.generator.gates.gemini import GeminiGate
    gate = get_llm_gate()
    if isinstance(gate, GeminiGate):
        return gate._get_client()
    # Fallback to creating a direct client
    g_gate = GeminiGate()
    return g_gate._get_client()

def generate_content_with_retry(prompt, is_image_list=False, temperature=0.0, system_instruction=None, response_mime_type=None):
    """
    Unified gatekeeper wrapper. Redirects directly to generate_via_gate
    for clean backward compatibility across the codebase.
    """
    from src.core.generator.gates import generate_via_gate
    return generate_via_gate(
        prompt=prompt,
        is_image_list=is_image_list,
        temperature=temperature,
        system_instruction=system_instruction,
        response_mime_type=response_mime_type
    )
