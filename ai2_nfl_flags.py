"""Strict conversion of explicit CSV verification flags.

Unknown or missing values are never considered verified.
"""
def verified_flag(value):
    if type(value) is bool:
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False
