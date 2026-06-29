from src.core.generator.gates.base import GeminiQuotaExhaustedError
from src.core.generator.gates.gemini import GeminiGate
from src.core.generator.gates.openrouter import OpenRouterGate

# Global controller state for dynamic gate routing
_active_gate_instance = None

def get_llm_gate():
    """Returns the current active LLM Gate instance."""
    global _active_gate_instance
    if _active_gate_instance is None:
        _active_gate_instance = GeminiGate()
    return _active_gate_instance

def switch_active_gate_to_openrouter():
    """Switches the global LLM Gate permanently to OpenRouter for this runtime session."""
    global _active_gate_instance
    if not isinstance(_active_gate_instance, OpenRouterGate):
        print("SYSTEM LOG: Switching active LLM gate to OpenRouter Gate.")
        _active_gate_instance = OpenRouterGate()

def generate_via_gate(prompt, is_image_list=False, temperature=0.3, system_instruction=None, response_mime_type=None):
    """
    Unified gatekeeper function that coordinates query execution across active gates.
    If the Gemini gate fails or reports quota exhaustion, it permanently switches to the OpenRouter gate.
    """
    gate = get_llm_gate()
    
    try:
        return gate.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            response_mime_type=response_mime_type,
            temperature=temperature
        )
    except (GeminiQuotaExhaustedError, Exception) as e:
        # If the gate is already OpenRouterGate, don't try to switch/retry
        if isinstance(gate, OpenRouterGate):
            raise e
            
        # Switch permanently to OpenRouter for subsequent requests
        print(f"Gemini Gate failed with exception: {e}. Switching active gate to OpenRouter...")
        switch_active_gate_to_openrouter()
        
        # Retry the failed request immediately using OpenRouterGate
        new_gate = get_llm_gate()
        try:
            return new_gate.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                response_mime_type=response_mime_type,
                temperature=temperature
            )
        except Exception as or_err:
            raise RuntimeError(
                f"Gemini Gate failed ({e}). OpenRouter Gate also failed: {or_err}"
            ) from e
