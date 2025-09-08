from fractions import Fraction
from collections import defaultdict
from pprint import pprint
from typing import Any, Iterable, TypeVar, cast
import numpy as np
import scipy.optimize
from src.utils import *
from src.config import *
from src.models import  *
from src.debug import Debugger
from src.xlsx_dump import XlxsDump
from src.args import get_parser
from src.data_parser import DataParser
import sys


T = TypeVar("T")


arg_parser = get_parser()
args = arg_parser.parse_args()


def float_to_clock(value: float) -> Fraction:
    return Fraction(round(value * INVERSE_CLOCK_GRANULARITY), INVERSE_CLOCK_GRANULARITY)


### Debug ###

dbug = Debugger()

if (args.dump_debug_info):
    dbug.init()

data_parser = DataParser()
data_parser.init(args, dbug)


### Configured clock speeds ###

MINER_CLOCKS = parse_clock_spec(args.miner_clocks)
MANUFACTURER_CLOCKS = parse_clock_spec(args.manufacturer_clocks)
SOMERSLOOP_CLOCKS = parse_clock_spec(args.somersloop_clocks)
GENERATOR_CLOCKS = parse_clock_spec(args.generator_clocks)

dbug.debug_dump(
    heading=
    "Configured clock speeds",
    obj=
    f"""
{MINER_CLOCKS=}
{MANUFACTURER_CLOCKS=}
{SOMERSLOOP_CLOCKS=}
{GENERATOR_CLOCKS=}
""".strip(),
)

### Configured resource multipliers ###

RESOURCE_MULTIPLIERS = parse_resource_multipliers(args.resource_multipliers)

dbug.debug_dump(
    "Configured resource multipliers",
    f"""
{RESOURCE_MULTIPLIERS=}
""".strip(),
)


### Configured disabled recipes ###

DISABLED_RECIPES: list[str] = [
    token.strip() for token in args.disabled_recipes.split(",")
]

dbug.debug_dump(
    "Configured disabled recipes",
    f"""
{DISABLED_RECIPES=}
""".strip(),
)


lp_columns: dict[str, LPColumn] = {}
lp_equalities: dict[str, float] = {}
lp_lower_bounds: dict[str, float] = {}


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


def add_lp_column(
    coeffs: dict[str, float],
    type_: str,
    name: str,
    display_name: str | None = None,
    machine_name: str | None = None,
    resource_subtype: str | None = None,
    clock: Fraction | None = None,
    somersloops: int | None = None,
    objective_weight: float | None = None,
    requires_integrality: bool = False,
):
    column_id = create_column_id(
        type_=type_,
        name=name,
        clock=clock,
        somersloops=somersloops,
    )
    display_name = display_name or name
    full_display_name = create_column_full_display_name(
        type_=type_,
        display_name=display_name,
        machine_name=machine_name,
        resource_subtype=resource_subtype,
        clock=clock,
        somersloops=somersloops,
    )
    assert column_id not in lp_columns, f"duplicate {column_id=}"
    lp_columns[column_id] = LPColumn(
        coeffs=coeffs,
        type_=type_,
        name=name,
        display_name=display_name,
        full_display_name=full_display_name,
        machine_name=machine_name,
        resource_subtype=resource_subtype,
        clock=clock,
        somersloops=somersloops,
        objective_weight=objective_weight,
        requires_integrality=requires_integrality,
    )






def add_miner_columns(resource: Resource):
    resource_id = resource.resource_id
    item_class = resource.item_class
    item = data_parser.data.items[item_class]

    miner = data_parser.get_miner_for_resource(resource)
    

    extraction_rate = miner.extraction_rate_base * resource.multiplier
    min_clock = miner.min_clock
    max_clock = data_parser.get_max_extraction_clock(miner, resource, extraction_rate)
    configured_clocks = MANUFACTURER_CLOCKS if resource.is_unlimited else MINER_CLOCKS


    clock_choices = data_parser.clamp_clock_choices(configured_clocks, min_clock, max_clock)

    resource_var = f"resource|{resource_id}"
    item_var = f"item|{item_class}"

    for clock in clock_choices:
        machines = 1 + (resource.num_satellites if resource.is_resource_well else 0)
        coeffs = {
            "machines": machines,
            "power_consumption": get_power_consumption(miner, clock),
            item_var: clock * extraction_rate,
        }

        if not resource.is_unlimited:
            coeffs[resource_var] = -1

        add_lp_column(
            coeffs,
            type_="miner",
            name=resource_id,
            display_name=item.display_name,
            machine_name=miner.display_name,
            resource_subtype=resource.subtype,
            clock=clock,
        )

    if not resource.is_unlimited:
        resource_multiplier = RESOURCE_MULTIPLIERS.get(
            resource.item_class,
            RESOURCE_MULTIPLIERS.get("All", RESOURCE_MULTIPLIERS.get("all", 1.0)),
        )
        lp_lower_bounds[resource_var] = -resource.count * resource_multiplier

    lp_equalities[item_var] = 0.0


for resource in data_parser.data.resources.values():
    add_miner_columns(resource)


def add_manufacturer_columns(recipe: Recipe):
    manufacturer_class = recipe.manufacturer
    manufacturer = data_parser.data.manufacturers[manufacturer_class]

    somersloop_choices: list[int | None] = [None]
    if manufacturer.can_change_production_boost:
        somersloop_choices.extend(range(1, manufacturer.production_shard_slot_size + 1))

    for somersloops in somersloop_choices:
        if somersloops is None:
            base_output_mult = 1.0
            base_power_mult = 1.0
            requires_integrality = False
        else:
            base_output_mult = 1.0 + somersloops * manufacturer.production_shard_boost_multiplier
            base_power_mult = base_output_mult ** manufacturer.production_boost_power_consumption_exponent
            requires_integrality = True

        # >>> New: global building multiplier applied to throughput and power <<<
        throughput_mult = base_output_mult * args.building_multiplier
        power_mult = base_power_mult * args.building_multiplier

        min_clock = manufacturer.min_clock
        max_clock = data_parser.get_max_recipe_clock(
            manufacturer, recipe, throughput_multiplier=throughput_mult  # changed name
        )
        configured_clocks = MANUFACTURER_CLOCKS if somersloops is None else SOMERSLOOP_CLOCKS
        clock_choices = data_parser.clamp_clock_choices(configured_clocks, min_clock, max_clock)

        for clock in clock_choices:
            power_consumption = get_power_consumption(manufacturer, clock, recipe) * power_mult
            coeffs = {"machines": 1, "power_consumption": power_consumption}
            if somersloops is not None:
                coeffs["somersloop_usage"] = somersloops

            recipe_coeffs = get_recipe_coeffs(
                recipe, clock=clock, throughput_multiplier=throughput_mult  # changed name
            )
            for item_var, coeff in recipe_coeffs.items():
                coeffs[item_var] = coeff
                lp_equalities[item_var] = 0.0

            add_lp_column(
                coeffs,
                type_="manufacturer",
                name=recipe.class_name,
                display_name=recipe.display_name,
                machine_name=manufacturer.display_name,
                clock=clock,
                somersloops=somersloops,
                requires_integrality=requires_integrality,
            )


for recipe in data_parser.data.recipes.values():
    if recipe.class_name not in DISABLED_RECIPES:
        add_manufacturer_columns(recipe)


def add_sink_column(item: Item):
    item_class = item.class_name
    item_var = f"item|{item_class}"
    points = item.points

    if not (item.form == "RF_SOLID" and points > 0):
        if args.allow_waste:
            add_lp_column(
                {item_var: -1},
                type_="waste",
                name=item_class,
                display_name=item.display_name,
            )
        return

    coeffs = {
        "machines": 1 / data_parser.constants.CONVEYOR_BELT_LIMIT,
        "power_consumption": data_parser.constants.SINK_POWER_CONSUMPTION / data_parser.constants.CONVEYOR_BELT_LIMIT,
        item_var: -1,
        "points": points,
    }

    add_lp_column(
        coeffs,
        type_="sink",
        name=item_class,
        display_name=item.display_name,
    )

    lp_equalities[item_var] = 0.0


for item in data_parser.data.items.values():
    add_sink_column(item)


def add_generator_columns(generator: PowerGenerator, fuel: Fuel):
    recipe = create_recipe_for_generator(generator, fuel, data_parser.data)
    fuel_item = data_parser.data.items[fuel.fuel_class]

    min_clock = generator.min_clock
    max_clock = data_parser.get_max_recipe_clock(generator, recipe)
    clock_choices = data_parser.clamp_clock_choices(GENERATOR_CLOCKS, min_clock, max_clock)

    for clock in clock_choices:
        power_production = (
            get_power_production(generator, clock=clock) * data_parser.constants.POWER_PRODUCTION_MULTIPLIER
        )
        coeffs = {
            "machines": 1,
            "power_production": power_production,
        }

        recipe_coeffs = get_recipe_coeffs(recipe, clock=clock)
        for item_var, coeff in recipe_coeffs.items():
            coeffs[item_var] = coeff
            lp_equalities[item_var] = 0.0

        add_lp_column(
            coeffs,
            type_="generator",
            name=fuel_item.class_name,
            display_name=fuel_item.display_name,
            machine_name=generator.display_name,
            clock=clock,
        )


for generator in data_parser.data.generators.values():
    for fuel in generator.fuels:
        add_generator_columns(generator, fuel)


def add_geothermal_generator_columns(resource: Resource):
    resource_id = resource.resource_id
    resource_var = f"resource|{resource_id}"

    power_production = (
        data_parser.data.geothermal_generator.mean_variable_power_production # type: ignore
        * resource.multiplier
        * data_parser.constants.POWER_PRODUCTION_MULTIPLIER
    )
    coeffs = {
        "machines": 1,
        "power_production": power_production,
        resource_var: -1,
    }

    add_lp_column(
        coeffs,
        type_="generator",
        name=resource_id,
        display_name=data_parser.get_item_display_name(GEOTHERMAL_CLASS),
        machine_name=data_parser.data.geothermal_generator.display_name, # type: ignore
        resource_subtype=resource.subtype,
        requires_integrality=True,
    )

    resource_multiplier = RESOURCE_MULTIPLIERS.get(
        resource.item_class,
        RESOURCE_MULTIPLIERS.get("All", RESOURCE_MULTIPLIERS.get("all", 1.0)),
    )

    lp_lower_bounds[resource_var] = -resource.count * resource_multiplier


for resource in data_parser.data.geysers.values():
    add_geothermal_generator_columns(resource)


def add_meta_coeffs(column_id: str, column: LPColumn):
    to_add: defaultdict[str, float] = defaultdict(float)
    for variable, coeff in column.coeffs.items():
        if variable.startswith("item|") and coeff > 0:
            item_class = variable[5:]
            if item_class not in data_parser.data.items:
                print(f"WARNING: item not found in items dict: {item_class}")
                continue

            item = data_parser.data.items[item_class]
            form = item.form
            conveyance_limit = data_parser.get_form_conveyance_limit(form)
            conveyance = coeff / conveyance_limit

            # Avoid incurring transport costs for Water Extractors,
            # as they would otherwise dominate the cost.
            # Basically we're assuming other stuff is brought to the water.
            if column.type_ == "miner" and column.resource_subtype != "extractor":
                to_add["transport_power_cost"] += args.transport_power_cost * conveyance
                to_add["drone_battery_cost"] += args.drone_battery_cost * conveyance

            if form == "RF_SOLID":
                to_add["conveyors"] += conveyance
            else:
                to_add["pipelines"] += conveyance

    for variable, coeff in to_add.items():
        if coeff != 0.0:
            column.coeffs[variable] = column.coeffs.get(variable, 0.0) + coeff


for column_id, column in lp_columns.items():
    add_meta_coeffs(column_id, column)


machine_limit = (
    HardLimit(name="machine_limit", weight=-1.0, lower_bound=-args.machine_limit)
    if args.machine_limit is not None
    else None
)


def add_objective_column(
    objective: str, objective_weight: float, hard_limit: HardLimit | None = None
):
    coeffs = {
        objective: -1.0,
    }
    if hard_limit is not None:
        coeffs[hard_limit.name] = hard_limit.weight
        lp_lower_bounds[hard_limit.name] = hard_limit.lower_bound
    add_lp_column(
        coeffs,
        type_="objective",
        name=objective,
        objective_weight=objective_weight,
    )
    lp_equalities[objective] = 0.0


add_objective_column("points", 1.0)
add_objective_column("machines", -args.machine_penalty, machine_limit)
add_objective_column("conveyors", -args.conveyor_penalty)
add_objective_column("pipelines", -args.pipeline_penalty)


# These columns cancel dummy variables introduced for ease of reporting.
# Instead of deducting X directly, we accumulate a cost variable, then pay it here.
# Breakdowns of cost contributors/payers then appear naturally in the report.

# Power usage
coeffs = {
    "power_consumption": -1.0,
    "power_production": -1.0,
}
add_lp_column(
    coeffs,
    type_="power",
    name="usage",
)
lp_equalities["power_consumption"] = 0.0
lp_lower_bounds["power_production"] = -data_parser.constants.ALIEN_POWER_AUGMENTER_TOTAL_STATIC_POWER
if args.infinite_power:
    lp_lower_bounds["power_production"] = -np.inf

# Alien Power Matrix fuel
if data_parser.constants.TOTAL_ALIEN_POWER_MATRIX_COST > 0.0:
    coeffs = {
        "alien_power_matrix_cost": -1.0,
        f"item|{ALIEN_POWER_MATRIX_CLASS}": -1.0,
    }
    add_lp_column(
        coeffs,
        type_="alien_power_matrix",
        name="fuel",
    )
    lp_equalities["alien_power_matrix_cost"] = -data_parser.constants.TOTAL_ALIEN_POWER_MATRIX_COST

# Configured extra costs
for extra_cost, cost_variable, cost_coeff in [
    ("transport_power_cost", "power_consumption", 1.0),
    ("drone_battery_cost", f"item|{BATTERY_CLASS}", -1.0),
]:
    coeffs = {
        extra_cost: -1.0,
        cost_variable: cost_coeff,
    }
    add_lp_column(
        coeffs,
        type_="extra_cost",
        name=extra_cost,
    )
    lp_equalities[extra_cost] = 0.0

# Somersloop usage
coeffs = {
    "somersloop_usage": -1.0,
    "somersloop": -1.0,
}
add_lp_column(
    coeffs,
    type_="somersloop",
    name="usage",
)
lp_equalities["somersloop_usage"] = 0.0
lp_lower_bounds["somersloop"] = -data_parser.constants.NUM_SOMERSLOOPS_AVAILABLE_FOR_PRODUCTION

# dbug.debug_dump("LP columns (before pruning)", lp_columns)
# dbug.debug_dump("LP equalities (before pruning)", lp_equalities)
# dbug.debug_dump("LP lower bounds (before pruning)", lp_lower_bounds)

# ########
# BATTERY_VAR = f"item|{BATTERY_CLASS}"
# print("=== PARSER SANITY ===")
# print("items:", len(data_parser.data.items))
# print("recipes:", len(data_parser.data.recipes))
# print("has battery item key:", BATTERY_VAR in {f"item|{k}" for k in data_parser.data.items.keys()})
# print("any column references BATTERY_VAR:", any(BATTERY_VAR in c.coeffs for c in lp_columns.values()))
# print("columns mentioning battery (pre-prune):", [cid for cid,c in lp_columns.items() if BATTERY_VAR in c.coeffs])

# Variables present before pruning
# def get_all_variables_snapshot():
#     vs = set()
#     for c in lp_columns.values():
#         vs.update(c.coeffs.keys())
#     return vs

# pre_vars = get_all_variables_snapshot()
# print("has variable 'drone_battery_cost' pre-prune:", "drone_battery_cost" in pre_vars)

#########




lp_variables = get_all_variables(lp_columns, lp_equalities, lp_lower_bounds)

# dbug.debug_dump("LP variables (before pruning)", lp_variables)


### Pruning unreachable items ###


reachable_items: set[str] = set()
while True:
    any_added = False
    for column_id, column in lp_columns.items():
        eligible = True
        to_add: set[str] = set()
        for variable, coeff in column.coeffs.items():
            if variable.startswith("item|") and variable not in reachable_items:
                if coeff > 0:
                    to_add.add(variable)
                elif coeff < 0:
                    eligible = False
                    break
        if eligible and to_add:
            any_added = True
            reachable_items |= to_add
    if not any_added:
        break

unreachable_items = (
    set(v for v in lp_variables if v.startswith("item|")) - reachable_items
)

dbug.debug_dump("Unreachable items to be pruned", unreachable_items)

columns_to_prune: list[str] = []
for column_id, column in lp_columns.items():
    for variable, coeff in column.coeffs.items():
        if variable in unreachable_items and coeff < 0:
            columns_to_prune.append(column_id)
            break
for column_id in columns_to_prune:
    del lp_columns[column_id]
for item_var in unreachable_items:
    if item_var in lp_equalities:
        del lp_equalities[item_var]


dbug.debug_dump("LP columns (after pruning)", lp_columns)
dbug.debug_dump("LP equalities (after pruning)", lp_equalities)
dbug.debug_dump("LP lower bounds (after pruning)", lp_lower_bounds)

lp_variables = get_all_variables(lp_columns, lp_equalities, lp_lower_bounds)

dbug.debug_dump("LP variables (after pruning)", lp_variables)


### LP run ###





# order is for report display, but we might as well sort it here
column_type_order = to_index_map(
    [
        "objective",
        "power",
        "extra_cost",
        "sink",
        "waste",
        "somersloop",
        "manufacturer",
        "miner",
        "generator",
    ]
)
resource_subtype_order = to_index_map(["pure", "normal", "impure"])
objective_order = to_index_map(
    ["points", "machines", "machine_limit", "conveyors", "pipelines"]
)
extra_cost_order = to_index_map(["transport_power_cost", "drone_battery_cost"])


def column_order_key(arg: tuple[str, LPColumn]):
    column_id, column = arg

    if column.type_ in column_type_order:
        type_key = (0, column_type_order[column.type_])
    else:
        type_key = (1, column.type_)

    name = column.display_name
    if column.type_ == "objective":
        name_key = objective_order[name]
    elif column.type_ == "extra_cost":
        name_key = extra_cost_order[name]
    else:
        name_key = name

    resource_subtype = column.resource_subtype
    if resource_subtype in resource_subtype_order:
        subtype_key = (0, resource_subtype_order[resource_subtype])
    else:
        subtype_key = (1, resource_subtype)

    return (type_key, name_key, subtype_key, column.clock, column_id)


sorted_columns = sorted(lp_columns.items(), key=column_order_key)
lp_variable_indices = to_index_map(lp_variables)

lp_c = np.zeros(len(lp_columns), dtype=np.double)
lp_integrality = np.zeros(len(lp_columns), dtype=np.int64)
lp_A = np.zeros((len(lp_variables), len(lp_columns)), dtype=np.double)
lp_b_l = np.zeros(len(lp_variables), dtype=np.double)
lp_b_u = np.zeros(len(lp_variables), dtype=np.double)

for column_index, (column_id, column) in enumerate(sorted_columns):
    if column.objective_weight is not None:
        lp_c[column_index] = column.objective_weight
    if column.requires_integrality:
        lp_integrality[column_index] = 1
    for variable, coeff in column.coeffs.items():
        lp_A[lp_variable_indices[variable], column_index] = coeff

for variable, rhs in lp_equalities.items():
    lp_b_l[lp_variable_indices[variable]] = rhs
    lp_b_u[lp_variable_indices[variable]] = rhs

for variable, rhs in lp_lower_bounds.items():
    lp_b_l[lp_variable_indices[variable]] = rhs
    lp_b_u[lp_variable_indices[variable]] = np.inf

lp_constraints = scipy.optimize.LinearConstraint(lp_A, lp_b_l, lp_b_u)  # type: ignore

print("LP running")

lp_result = scipy.optimize.milp(
    -lp_c,
    integrality=lp_integrality,
    constraints=lp_constraints,
)

if lp_result.status != 0:
    print("ERROR: LP did not terminate successfully")
    pprint(lp_result)
    sys.exit(1)

print("LP result:")
pprint(lp_result)


### Display formatting ###


REPORT_EPSILON = 1e-7

column_results: list[tuple[str, LPColumn, float]] = [
    (column_id, column, lp_result.x[column_index])
    for column_index, (column_id, column) in enumerate(sorted_columns)
]

if not args.show_unused:
    column_results = list(filter(lambda x: abs(x[2]) > REPORT_EPSILON, column_results))





variable_type_order = to_index_map(
    from_index_map(objective_order)
    + ["power_production", "power_consumption"]
    + ["alien_power_matrix_cost"]
    + from_index_map(extra_cost_order)
    + ["somersloop", "somersloop_usage"]
    + ["item", "resource"]
)


def create_empty_variable_breakdown(variable: str) -> VariableBreakdown:
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


variable_breakdowns = {
    variable: create_empty_variable_breakdown(variable) for variable in lp_variables
}

lp_objective: float = -lp_result.fun

print("")
print("Summary:")

print(f"{lp_objective:>17.3f} objective")

for column_id, column, column_coeff in column_results:
    print(f"{column_coeff:>17.3f} {column.full_display_name}")

    for variable, coeff in column.coeffs.items():
        rate = column_coeff * coeff
        budget_entry = BudgetEntry(
            desc=column.full_display_name,
            count=column_coeff,
            rate=abs(rate),
            share=0.0,
        )
        if abs(rate) < REPORT_EPSILON:
            continue
        elif rate > 0:
            variable_breakdowns[variable].production.append(budget_entry)
        else:
            variable_breakdowns[variable].consumption.append(budget_entry)

print("")


def finalize_variable_budget_side(budget_side: list[BudgetEntry]):
    if not budget_side:
        return
    total_rate = 0.0
    total_count = 0.0
    for budget_entry in budget_side:
        total_rate += budget_entry.rate
        total_count += budget_entry.count
    for budget_entry in budget_side:
        budget_entry.share = budget_entry.rate / total_rate
    budget_side.sort(key=lambda entry: (-entry.share, entry.desc))
    total = BudgetEntry(desc="Total", count=total_count, rate=total_rate, share=1.0)
    budget_side.insert(0, total)


lp_Ax = lp_A @ lp_result.x

for variable, breakdown in variable_breakdowns.items():
    # don't show offsetting dummy items in the breakdown (e.g. "objective|points" as consumer of points)
    # currently these are precisely the consumption of special variables, but that may change
    if breakdown.type_ not in ["item", "resource"]:
        breakdown.consumption = []

    variable_index = lp_variable_indices[variable]
    if variable in lp_lower_bounds:
        slack: float = lp_Ax[variable_index] - lp_b_l[variable_index]
        if slack < -REPORT_EPSILON:
            print(f"WARNING: lower bound violation: {variable=} {slack=}")
        breakdown.initial = -lp_lower_bounds[variable]
        breakdown.final = slack if abs(slack) > REPORT_EPSILON else 0
    else:
        residual: float = lp_Ax[variable_index] - lp_b_l[variable_index]
        if abs(residual) > REPORT_EPSILON:
            print(f"WARNING: equality constraint violation: {variable=} {residual=}")
    finalize_variable_budget_side(breakdown.production)
    finalize_variable_budget_side(breakdown.consumption)

sorted_variable_breakdowns = sorted(
    variable_breakdowns.values(), key=lambda bd: bd.sort_key
)

if args.xlsx_report:
    writer = XlxsDump()
    writer.define(
        column_results=column_results,
        lp_objective=lp_objective,
        sorted_variable_breakdowns=sorted_variable_breakdowns
                  )
    writer.dump(args)

