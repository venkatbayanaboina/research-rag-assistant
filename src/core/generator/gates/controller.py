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
    import time
    from src.core.utils.profiler import log_timing
    
    gate = get_llm_gate()
    gate_name = gate.__class__.__name__
    
    start_time = time.time()
    try:
        res = gate.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            response_mime_type=response_mime_type,
            temperature=temperature
        )
        duration = time.time() - start_time
        log_timing(
            step_name="llm_answer_generation",
            duration_seconds=duration,
            metadata={
                "provider_gate": gate_name,
                "is_fallback": False
            }
        )
        return res
    except (GeminiQuotaExhaustedError, Exception) as e:
        # If the gate is already OpenRouterGate, don't try to switch/retry
        if isinstance(gate, OpenRouterGate):
            duration = time.time() - start_time
            log_timing(
                step_name="llm_answer_generation_failed",
                duration_seconds=duration,
                metadata={
                    "provider_gate": gate_name,
                    "error": str(e)
                }
            )
            raise e
            
        # Switch permanently to OpenRouter for subsequent requests
        print(f"Gemini Gate failed with exception: {e}. Switching active gate to OpenRouter...")
        switch_active_gate_to_openrouter()
        
        # Retry the failed request immediately using OpenRouterGate
        new_gate = get_llm_gate()
        new_gate_name = new_gate.__class__.__name__
        
        fallback_start_time = time.time()
        try:
            res = new_gate.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                response_mime_type=response_mime_type,
                temperature=temperature
            )
            duration = time.time() - fallback_start_time
            log_timing(
                step_name="fallback_answer_generation",
                duration_seconds=duration,
                metadata={
                    "provider_gate": new_gate_name,
                    "is_fallback": True
                }
            )
            return res
        except Exception as or_err:
            duration = time.time() - fallback_start_time
            log_timing(
                step_name="fallback_answer_generation_failed",
                duration_seconds=duration,
                metadata={
                    "provider_gate": new_gate_name,
                    "error": str(or_err)
                }
            )
            raise RuntimeError(
                f"Gemini Gate failed ({e}). OpenRouter Gate also failed: {or_err}"
            ) from e
