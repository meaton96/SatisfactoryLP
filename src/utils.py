#!/usr/bin/env python
# pyright: basic

from fractions import Fraction
from src.config import INVERSE_CLOCK_GRANULARITY
from .models import Consts, PowerConsumer, PowerGenerator, Recipe, Fuel, Data
from typing import Iterable

def float_to_clock(value: float) -> Fraction:
    return Fraction(round(value * INVERSE_CLOCK_GRANULARITY), INVERSE_CLOCK_GRANULARITY)


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