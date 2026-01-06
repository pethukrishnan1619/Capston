"""
Tool implementations: weather and calculator.
This module defines input schemas using pydantic and exposes functions to call external APIs
or perform computations. Functions return a dictionary with an 'ok' flag and either 'result'
or 'error' details.
"""
import re
import requests
from typing import Dict, Any
from pydantic import BaseModel, Field, ValidationError

__all__ = [
    "WeatherToolInput",
    "weather_tool_call",
    "CalculationInput",
    "calculator_tool_call",
]


class WeatherToolInput(BaseModel):
    """Input schema for the weather tool."""
    location: str = Field(..., description="City name like Chennai, Mumbai, London")
    days: int = Field(3, ge=1, le=7, description="Forecast days (1 to 7)")


def weather_tool_call(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the Open‑Meteo API to fetch a weather forecast for the given location and number of days.

    Returns a dict with:
        ok (bool): indicates success
        location (str): resolved location name
        forecast_days (int): number of days requested
        forecast (list): list of daily forecast dicts with date, temp_max_c, temp_min_c,
                          precip_mm and wind_max_kmh
    On failure, returns ok=False and an error message.
    """
    try:
        inp = WeatherToolInput(**data)
    except ValidationError as e:
        return {"ok": False, "error": str(e)}

    loc = inp.location.strip()
    if not loc:
        return {"ok": False, "error": "Location cannot be empty."}

    # Step 1: Geocode to get latitude and longitude
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        gr = requests.get(geo_url, params={"name": loc, "count": 1}, timeout=15)
        gr.raise_for_status()
        gj = gr.json()
    except Exception as e:
        return {"ok": False, "error": f"Geocoding failed: {e}"}

    if not gj.get("results"):
        return {"ok": False, "error": f"Location not found: {loc}"}

    place = gj["results"][0]
    lat, lon = place["latitude"], place["longitude"]

    # Step 2: Fetch forecast
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "forecast_days": inp.days,
        "timezone": "auto",
    }
    try:
        fr = requests.get(forecast_url, params=params, timeout=20)
        fr.raise_for_status()
        fj = fr.json()
    except Exception as e:
        return {"ok": False, "error": f"Forecast failed: {e}"}

    daily = fj.get("daily", {})
    forecast = []
    n = len(daily.get("time", []))
    for i in range(n):
        forecast.append({
            "date": daily["time"][i],
            "temp_max_c": daily["temperature_2m_max"][i],
            "temp_min_c": daily["temperature_2m_min"][i],
            "precip_mm": daily["precipitation_sum"][i],
            "wind_max_kmh": daily["wind_speed_10m_max"][i],
        })

    return {
        "ok": True,
        "location": f"{place.get('name')}, {place.get('country')}",
        "forecast_days": inp.days,
        "forecast": forecast,
    }


# class CalculationInput(BaseModel):
#     """Input schema for the calculator tool."""
#     expression: str = Field(..., description="Math expression e.g. (10+20)/2")

# # Allowed characters for the safe eval
# _ALLOWED_EXPR = re.compile(r"^[0-9\.\+\-\*/\(\)\s]+$")


# def calculator_tool_call(data: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Safely evaluate a simple arithmetic expression.

#     Returns a dict with ok=True and result on success, or ok=False with an error message on failure.
#     """
#     try:
#         inp = CalculationInput(**data)
#     except ValidationError as e:
#         return {"ok": False, "error": str(e)}

#     expr = inp.expression.strip()
#     if not _ALLOWED_EXPR.fullmatch(expr):
#         return {"ok": False, "error": "Invalid characters. Only numbers and + - * / ( ) . allowed."}

#     try:
#         # Evaluate the expression using a restricted global environment
#         result = eval(expr, {"__builtins__": {}}, {})
#         return {"ok": True, "result": result}
#     except Exception as e:
#         return {"ok": False, "error": f"Calculation failed: {e}"}


# At the top of tools.py, add this import:
import math

# Existing CalculationInput can stay the same.
class CalculationInput(BaseModel):
    expression: str = Field(
        ..., description="Math expression, e.g. sin(pi/2) + sqrt(16) - 3**2"
    )

def calculator_tool_call(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate an arithmetic or scientific expression using a restricted set of
    operators and functions. Supported operations include +, -, *, /, %, **,
    and functions such as sqrt, log, sin, cos, tan, exp, factorial, as well as
    the constants pi and e. Returns {'ok': True, 'result': value} on success,
    otherwise {'ok': False, 'error': ...}.
    """
    try:
        inp = CalculationInput(**data)
    except ValidationError as e:
        return {"ok": False, "error": str(e)}

    expr = inp.expression.strip()
    if not expr:
        return {"ok": False, "error": "Expression cannot be empty."}

    # Define allowed names mapping to math functions and constants
    allowed_names: Dict[str, Any] = {
        "abs": abs,
        "pow": pow,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sqrt": math.sqrt,
        "log": math.log,       # natural log
        "log10": math.log10,   # base-10 log
        "exp": math.exp,
        "factorial": math.factorial,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        # Evaluate the expression safely with no builtins and only allowed names
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": f"Calculation failed: {e}"}
