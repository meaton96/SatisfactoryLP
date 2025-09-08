from __future__ import annotations
### Classes ###

from dataclasses import dataclass, field

from fractions import Fraction
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.sparse import csc_matrix
from scipy.optimize import LinearConstraint
import numpy as np
from numpy.typing import NDArray
import re


@dataclass(slots=True)
class LPResults:
    lp_result: Optional[OptimizeResult] = None
    sorted_columns: Optional[List[Tuple[str, "LPColumn"]]] = None
    objective_order: Dict[str, int] = field(default_factory=dict)
    extra_cost_order: Dict[str, int] = field(default_factory=dict)
    resource_subtype_order: Dict[str, int] = field(default_factory=dict)
    lp_variables: Set[str] = field(default_factory=set)
    lp_A: Optional[NDArray[np.float64]] = None
    lp_b_l: Optional[NDArray[np.float64]] = None
    lp_b_u: Optional[NDArray[np.float64]] = None
    lp_variable_indices: Dict[str, int] = field(default_factory=dict)
    lp_lower_bounds: Dict[str, float] = field(default_factory=dict)

@dataclass(slots=True)
class LPClocks:
    MINER_CLOCKS: List[Fraction] = field(default_factory=list)
    MANUFACTURER_CLOCKS: List[Fraction] = field(default_factory=list)
    SOMERSLOOP_CLOCKS: List[Fraction] = field(default_factory=list)
    GENERATOR_CLOCKS: List[Fraction] = field(default_factory=list)

@dataclass(slots=True)
class LPConfig:
    lp_clocks: LPClocks = field(default_factory=LPClocks)
    resource_multipliers: Dict[str, float] = field(default_factory=dict)
    disabled_recipes: List[str] = field(default_factory=list)
    lp_columns: Dict[str, "LPColumn"] = field(default_factory=dict)
    lp_equalities: Dict[str, float] = field(default_factory=dict)
    lp_lower_bounds: Dict[str, float] = field(default_factory=dict)
    lp_c: Optional[NDArray[np.float64]] = None
    lp_constraints: Optional[LinearConstraint] = None
    lp_integrality: Optional[NDArray[np.int64]] = None



@dataclass(slots=True)
class Data:
    items: Dict[str, Item] = field(default_factory=dict)
    miners: Dict[str, Miner] = field(default_factory=dict)
    resources: Dict[str, Resource] = field(default_factory=dict)
    geysers: Dict[str, Resource] = field(default_factory=dict)
    manufacturers: Dict[str, Manufacturer] = field(default_factory=dict)
    recipes: Dict[str, Recipe] = field(default_factory=dict)
    generators: Dict[str, PowerGenerator] = field(default_factory=dict)
    map_info: Dict[str, Any] = field(default_factory=dict)
    geothermal_generator: Optional[GeothermalGenerator] = None



@dataclass(slots=True)
class Consts:
    POWER_PRODUCTION_MULTIPLIER: float = 0.0
    CONVEYOR_BELT_LIMIT: float = 0.0
    PIPELINE_LIMIT: float = 0.0
    SINK_POWER_CONSUMPTION: float = 0.0
    NUM_SOMERSLOOPS_AVAILABLE: float = 0.0
    NUM_SOMERSLOOPS_AVAILABLE_FOR_PRODUCTION: float = 0.0
    ALIEN_POWER_AUGMENTER_TOTAL_STATIC_POWER: float = 0.0
    ALIEN_POWER_AUGMENTER_TOTAL_CIRCUIT_BOOST: float = 0.0
    TOTAL_ALIEN_POWER_MATRIX_COST: float = 0.0

    QUALIFIED_CLASS_NAME_REGEX: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"\"?/Script/[^']+'/[\w\-/]+\.(\w+)'\"?"),
        init=False
    )
    UNQUALIFIED_CLASS_NAME_REGEX: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"\"?/[\w\-/]+\.(\w+)\"?"),
        init=False
    )
    ITEM_AMOUNT_REGEX: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"\(ItemClass=([^,]+),Amount=(\d+)\)"),
        init=False
    )


# game modeling
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
    mean_variable_power_production: float = 0.0

@dataclass
class Resource:
    resource_id: str
    item_class: str
    subtype: str
    multiplier: float
    is_unlimited: bool
    count: int
    is_resource_well: bool
    num_satellites: int

# Linear Programming Setup
@dataclass
class LPColumn:
    coeffs: dict[str, float]
    type_: str
    name: str
    display_name: str
    full_display_name: str
    machine_name: str | None
    resource_subtype: str | None
    clock: Fraction | None
    somersloops: int | None
    objective_weight: float | None
    requires_integrality: bool

@dataclass
class BudgetEntry:
    desc: str
    count: float
    rate: float
    share: float


@dataclass
class VariableBreakdown:
    type_: str
    display_name: str
    sort_key: Any
    production: list[BudgetEntry]
    consumption: list[BudgetEntry]
    initial: float | None
    final: float | None

@dataclass
class HardLimit:
    name: str
    weight: float
    lower_bound: float