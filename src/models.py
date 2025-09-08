### Classes ###

from dataclasses import dataclass

from fractions import Fraction

@dataclass
class ClassObject:
    class_name: str
    display_name: str


@dataclass
class Machine(ClassObject):
    min_clock: Fraction
    max_clock: Fraction


@dataclass
class PowerConsumer(Machine):
    power_consumption: float
    power_consumption_exponent: float
    is_variable_power: bool


@dataclass
class Miner(PowerConsumer):
    extraction_rate_base: float
    uses_resource_wells: bool
    allowed_resource_forms: list[str]
    only_allow_certain_resources: bool
    allowed_resources: list[str] | None

    def check_allowed_resource(self, item_class: str, form: str) -> bool:
        if form not in self.allowed_resource_forms:
            return False
        if self.only_allow_certain_resources:
            assert self.allowed_resources
            return item_class in self.allowed_resources
        else:
            return True


@dataclass
class Manufacturer(PowerConsumer):
    can_change_production_boost: bool
    base_production_boost: float
    production_shard_slot_size: int
    production_shard_boost_multiplier: float
    production_boost_power_consumption_exponent: float


@dataclass
class Recipe(ClassObject):
    manufacturer: str
    inputs: list[tuple[str, float]]
    outputs: list[tuple[str, float]]
    mean_variable_power_consumption: float


@dataclass
class Item(ClassObject):
    class_name: str
    display_name: str
    form: str
    points: int
    stack_size: int
    energy: float


@dataclass
class Fuel:
    fuel_class: str
    supplemental_resource_class: str | None
    byproduct: str | None
    byproduct_amount: int


@dataclass
class PowerGenerator(Machine):
    fuels: list[Fuel]
    power_production: float
    requires_supplemental: bool
    supplemental_to_power_ratio: float


@dataclass
class GeothermalGenerator(Machine):
    mean_variable_power_production: float
