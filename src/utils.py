#!/usr/bin/env python
# pyright: basic

from fractions import Fraction
from src.config import INVERSE_CLOCK_GRANULARITY, ADDITIONAL_DISPLAY_NAMES
from .models import *
from typing import Iterable, cast, TypeVar
from collections import defaultdict

T = TypeVar("T")



def str_to_clock(s: str) -> Fraction:
    return float_to_clock(float(s))


def clock_to_percent_str(clock: Fraction) -> str:
    return f"{float(100 * clock)}%"

def parse_paren_list(s: str) -> list[str] | None:
        if not s:
            return None
        assert s.startswith("(") and s.endswith(")")
        s = s[1:-1]
        if not s:
            return []
        else:
            return s.split(",")
        
def extract_class_name(s: str, consts: Consts) -> str:
        m = consts.QUALIFIED_CLASS_NAME_REGEX.fullmatch(
            s
        ) or consts.UNQUALIFIED_CLASS_NAME_REGEX.fullmatch(s)
        assert m is not None, s
        return m.group(1)

def parse_class_list(s: str, constants: Consts) -> list[str] | None:
        l = parse_paren_list(s)
        if l is None:
            return None
        return [extract_class_name(x, constants) for x in l]

def get_power_production(generator: PowerGenerator, clock: Fraction) -> float:
        return generator.power_production * clock

def get_power_consumption(
        machine: PowerConsumer, clock: Fraction, recipe: Recipe | None = None
    ) -> float:
        power_consumption = machine.power_consumption
        if recipe is not None and machine.is_variable_power:
            power_consumption += recipe.mean_variable_power_consumption
        return power_consumption * (clock**machine.power_consumption_exponent)

def find_item_amounts(s: str, constants: Consts) -> Iterable[tuple[str, int]]:
        for m in constants.ITEM_AMOUNT_REGEX.finditer(s):
            yield (extract_class_name(m[1], constants), int(m[2]))


# It's convenient to consider generators burning fuel as recipes,
    # even though they are not actually listed as recipes.
def create_recipe_for_generator(generator: PowerGenerator, fuel: Fuel, data: Data) -> Recipe:
    inputs: list[tuple[str, float]] = []
    outputs: list[tuple[str, float]] = []

    power_production = generator.power_production

    fuel_class = fuel.fuel_class
    fuel_item = data.items[fuel_class]
    fuel_rate = 60.0 * power_production / fuel_item.energy
    inputs.append((fuel_class, fuel_rate))

    if generator.requires_supplemental:
        assert fuel.supplemental_resource_class is not None
        supplemental_class = fuel.supplemental_resource_class
        supplemental_rate = (
            60.0 * power_production * generator.supplemental_to_power_ratio
        )
        inputs.append((supplemental_class, supplemental_rate))

    if fuel.byproduct is not None:
        byproduct_class = fuel.byproduct
        byproduct_rate = fuel_rate * fuel.byproduct_amount
        outputs.append((byproduct_class, byproduct_rate))

    return Recipe(
        class_name="",
        display_name="",
        manufacturer=generator.class_name,
        inputs=inputs,
        outputs=outputs,
        mean_variable_power_consumption=0.0,
    )

def float_to_clock(value: float) -> Fraction:
    return Fraction(round(value * INVERSE_CLOCK_GRANULARITY), INVERSE_CLOCK_GRANULARITY)

def parse_clock_spec(s: str) -> list[Fraction]:
    result: list[Fraction] = []
    for token in s.split(","):
        token = token.strip()
        if "/" in token:
            bounds, _, step_str = token.rpartition("/")
            lower_str, _, upper_str = bounds.rpartition("-")
            lower = str_to_clock(lower_str)
            upper = str_to_clock(upper_str)
            step = str_to_clock(step_str)
            current = lower
            while current <= upper:
                result.append(current)
                current += step
        else:
            result.append(str_to_clock(token))
    result.sort()
    return result

def parse_resource_multipliers(s: str) -> dict[str, float]:
    result: dict[str, float] = {}
    if s:
        for token in s.split(","):
            item_class, _, multiplier = token.partition(":")
            item_class = item_class.strip()
            assert item_class not in result
            result[item_class] = float(multiplier)
    return result

def to_index_map(seq: Iterable[T]) -> dict[T, int]:
    return {value: index for index, value in enumerate(seq)}


def from_index_map(d: dict[T, int]) -> list[T]:
    result: list[T | None] = [None] * len(d)
    for value, index in d.items():
        result[index] = value
    assert all(value is not None for value in result)
    return cast(list[T], result)

def get_recipe_coeffs(
                recipe: Recipe, clock: Fraction, throughput_multiplier: float = 1.0
            ) -> defaultdict[str, float]:
    coeffs: defaultdict[str, float] = defaultdict(float)

    for item_class, input_rate in recipe.inputs:
        item_var = f"item|{item_class}"
        coeffs[item_var] -= clock * input_rate * throughput_multiplier

    for item_class, output_rate in recipe.outputs:
        item_var = f"item|{item_class}"
        coeffs[item_var] += clock * output_rate * throughput_multiplier

    return coeffs

def get_all_variables(
          lp_columns: dict[str, LPColumn],
          lp_equalities: dict[str, float],
          lp_lower_bounds: dict[str, float]) -> set[str]:
    variables: set[str] = set()

    for column in lp_columns.values():
        for variable in column.coeffs.keys():
            variables.add(variable)

    for variable in variables:
        if variable not in lp_equalities and variable not in lp_lower_bounds:
            print(f"WARNING: no constraint for variable: {variable}")

    for variable in lp_equalities.keys():
        if variable not in variables:
            print(f"INFO: equality constraint with unknown variable: {variable}")

    for variable in lp_lower_bounds.keys():
        if variable not in variables:
            print(f"WARNING: lower bound constraint with unknown variable: {variable}")

    return variables



def clamp_clock_choices(
        configured_clocks: list[Fraction], min_clock: Fraction, max_clock: Fraction
    ) -> list[Fraction]:
        

        assert min_clock < max_clock
        return sorted(
            {min(max_clock, max(min_clock, clock)) for clock in configured_clocks}
        )


    


def create_empty_variable_breakdown(variable: str,
                                    data_parser: Any,
                                    variable_type_order: dict[Any, int],
                                    resource_subtype_order: Any
                                    ) -> VariableBreakdown:
    tokens = variable.split("|")
    type_ = tokens[0]
    if type_ == "item" or type_ == "resource":
        item_class = tokens[1]
        tokens[1] = data_parser.get_item_display_name(item_class)
    display_name = "|".join(tokens)
    sort_key: list[Any] = [variable_type_order[type_]]
    if type_ == "resource":
        sort_key.append(tokens[1])
        sort_key.append(resource_subtype_order.get(tokens[2], np.inf))
        sort_key.append(tokens[2])
    else:
        sort_key.append(display_name)
    return VariableBreakdown(
        type_=type_,
        display_name=display_name,
        sort_key=sort_key,
        production=[],
        consumption=[],
        initial=None,
        final=None,
    )