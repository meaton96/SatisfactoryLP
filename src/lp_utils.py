from .utils import *

def create_column_id(
    type_: str,
    name: str,
    clock: Fraction | None = None,
    somersloops: int | None = None,
):
    tokens = [type_, name]
    if clock is not None:
        tokens.append(clock_to_percent_str(clock))
    if somersloops is not None:
        tokens.append(f"S:{somersloops}")
    return "|".join(tokens)

def create_column_full_display_name(
    type_: str,
    display_name: str,
    machine_name: str | None,
    resource_subtype: str | None,
    clock: Fraction | None = None,
    somersloops: int | None = None,
):
    tokens = [machine_name or type_, display_name]
    if resource_subtype is not None:
        tokens.append(resource_subtype)
    if clock is not None:
        tokens.append(clock_to_percent_str(clock))
    if somersloops is not None:
        tokens.append(f"S:{somersloops}")
    return "|".join(tokens)