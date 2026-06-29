from src.core.generator.gates.base import GeminiQuotaExhaustedError
from src.core.generator.gates.gemini import GeminiGate
from src.core.generator.gates.openrouter import OpenRouterGate
from src.core.generator.gates.ollama import OllamaGate

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

def switch_active_gate_to_ollama():
    """Switches the global LLM Gate permanently to local Ollama for this runtime session."""
    global _active_gate_instance
    if not isinstance(_active_gate_instance, OllamaGate):
        print("SYSTEM LOG: Switching active LLM gate to local Ollama Gate.")
        _active_gate_instance = OllamaGate()

def generate_via_gate(prompt, is_image_list=False, temperature=0.3, system_instruction=None, response_mime_type=None):
    """
    Unified gatekeeper function that coordinates query execution across active gates.
    Failover chain: Gemini -> OpenRouter -> Ollama (local)
    """
    import time
    from src.core.utils.profiler import log_timing
    
    # Get current active gate
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
    except Exception as e:
        print(f"⚠️ Active Gate {gate_name} failed: {e}")
        
        # Fallback Level 1: If Gemini failed, try OpenRouter
        if isinstance(gate, GeminiGate):
            print("Switching active gate to OpenRouter...")
            switch_active_gate_to_openrouter()
            gate = get_llm_gate()
            gate_name = gate.__class__.__name__
            
            fallback_start_time = time.time()
            try:
                res = gate.generate(
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
                        "provider_gate": gate_name,
                        "is_fallback": True
                    }
                )
                return res
            except Exception as or_err:
                print(f"⚠️ Fallback OpenRouter Gate failed: {or_err}")
                e = or_err # Update error context to the OpenRouter failure
                
        # Fallback Level 2: If OpenRouter failed (or we are already on OpenRouter), try local Ollama
        if isinstance(gate, OpenRouterGate):
            print("Switching active gate to local Ollama...")
            switch_active_gate_to_ollama()
            gate = get_llm_gate()
            gate_name = gate.__class__.__name__
            
            ollama_start_time = time.time()
            try:
                res = gate.generate(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    response_mime_type=response_mime_type,
                    temperature=temperature
                )
                duration = time.time() - ollama_start_time
                log_timing(
                    step_name="fallback_answer_generation",
                    duration_seconds=duration,
                    metadata={
                        "provider_gate": gate_name,
                        "is_fallback": True
                    }
                )
                return res
            except Exception as ollama_err:
                duration = time.time() - ollama_start_time
                log_timing(
                    step_name="fallback_answer_generation_failed",
                    duration_seconds=duration,
                    metadata={
                        "provider_gate": gate_name,
                        "error": str(ollama_err)
                    }
                )
                raise RuntimeError(
                    f"All gates failed. Gemini/OpenRouter failed, and local Ollama also failed: {ollama_err}"
                ) from ollama_err
                
        # If we are already on Ollama and it failed
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
