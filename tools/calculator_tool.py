from langchain_core.tools import tool
import math

@tool
def calculate(expression: str) -> str:
    """
    Mathematical calculations karne ke liye.
    Basic math (2+2), percentages (15% of 200), 
    aur scientific calculations (sqrt, log, etc) ke liye use karo.
    Example inputs: '2+2', 'sqrt(144)', '15/100*200'
    """
    try:
        # Safe math functions allow karo
        allowed_names = {
            'sqrt': math.sqrt,
            'log': math.log,
            'log10': math.log10,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e,
            'abs': abs,
            'round': round,
            'pow': pow
        }
        
        # Expression evaluate karo
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        
        return f"Calculation: {expression} = {result}"
        
    except Exception as e:
        return f"Calculation mein error: {str(e)}. Valid expression do jaise '2+2' ya 'sqrt(16)'"