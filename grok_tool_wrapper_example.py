# How to wrap any Grok tool call with resonance pruning
from resonance_dissonance import dissonance

def safe_tool_call(tool_func, *args, **kwargs):
    result = tool_func(*args, **kwargs)
    
    # Example: score factual dissonance against known invariants
    if "energy" in str(result).lower():
        T_val = some_energy_check(result)
        diss = dissonance(T_val)
        if diss > 3.0:
            print("High dissonance → triggering critique agent")
            # → spawn multi-agent review (FAN1.3 style)
    return result
