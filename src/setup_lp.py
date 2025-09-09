from .utils import *
from typing import Any
from .lp_utils import *
from .data_parser import DataParser
from .debug import Debugger
from .config import GEOTHERMAL_CLASS, ALIEN_POWER_MATRIX_CLASS, BATTERY_CLASS
import numpy as np
import scipy.optimize
import sys
from pprint import pprint

class SetupLP:

    

    def __init__(self) -> None:
        self.config: LPConfig
        self._data: DataParser
        self._dbug: Debugger
        self._args: Any
        self.result: LPResults
        


    def configure(self, args: Any, dbug: Any, data: DataParser):

        self._data = data
        self._dbug = dbug
        self._args = args
        self.config = LPConfig()
        self.result = LPResults()
        ### Configured clock speeds ###
        self.config.lp_clocks.MINER_CLOCKS = parse_clock_spec(args.miner_clocks)
        self.config.lp_clocks.MANUFACTURER_CLOCKS = parse_clock_spec(args.manufacturer_clocks)
        self.config.lp_clocks.SOMERSLOOP_CLOCKS = parse_clock_spec(args.somersloop_clocks)
        self.config.lp_clocks.GENERATOR_CLOCKS = parse_clock_spec(args.generator_clocks)

        dbug.debug_dump(
            heading=
            "Configured clock speeds",
            obj=
            f"""
        {self.config.lp_clocks.MINER_CLOCKS=}
        {self.config.lp_clocks.MANUFACTURER_CLOCKS=}
        {self.config.lp_clocks.SOMERSLOOP_CLOCKS=}
        {self.config.lp_clocks.GENERATOR_CLOCKS=}
        """.strip(),
        )

        ### Configured resource multipliers ###

        self.config.resource_multipliers = parse_resource_multipliers(args.resource_multipliers)

        dbug.debug_dump(
            "Configured resource multipliers",
            f"""
        {self.config.resource_multipliers=}
        """.strip(),
        )

        ### Configured disabled recipes ###

        self.config.disabled_recipes = [
            token.strip() for token in args.disabled_recipes.split(",")
        ]

        dbug.debug_dump(
            "Configured disabled recipes",
            f"""
        {self.config.disabled_recipes=}
        """.strip(),
        )

    def run_setup(self):
        # miners
        for resource in self._data.data.resources.values():
            self.add_miner_columns(resource)

        # producers
        for recipe in self._data.data.recipes.values():
            if recipe.class_name not in self.config.disabled_recipes:
                self.add_manufacturer_columns(recipe)

        # sink column
        for item in self._data.data.items.values():
                self.add_sink_column(item)

        # generators
        for generator in self._data.data.generators.values():
            for fuel in generator.fuels:
                self.add_generator_columns(generator, fuel)

        # Geothermal
        for resource in self._data.data.geysers.values():
            self.add_geothermal_generator_columns(resource)

        # Coeffecients 
        for column_id, column in self.config.lp_columns.items():
            self.add_meta_coeffs(column_id, column)

        machine_limit = (
            HardLimit(name="machine_limit", weight=-1.0, lower_bound=-self._args.machine_limit)
            if self._args.machine_limit is not None
            else None
        )

        self.add_objective_column("points", 1.0)
        self.add_objective_column("machines", -self._args.machine_penalty, machine_limit)
        self.add_objective_column("conveyors", -self._args.conveyor_penalty)
        self.add_objective_column("pipelines", -self._args.pipeline_penalty)

        # These columns cancel dummy variables introduced for ease of reporting.
        # Instead of deducting X directly, we accumulate a cost variable, then pay it here.
        # Breakdowns of cost contributors/payers then appear naturally in the report.

        # Power usage
        coeffs = {
            "power_consumption": -1.0,
            "power_production": -1.0,
        }
        self.add_lp_column(
            coeffs,
            type_="power",
            name="usage",
        )
        self.config.lp_equalities["power_consumption"] = 0.0
        self.config.lp_lower_bounds["power_production"] = -self._data.constants.ALIEN_POWER_AUGMENTER_TOTAL_STATIC_POWER
        if self._args.infinite_power:
            self.config.lp_lower_bounds["power_production"] = -np.inf

        # Alien Power Matrix fuel
        if self._data.constants.TOTAL_ALIEN_POWER_MATRIX_COST > 0.0:
            coeffs = {
                "alien_power_matrix_cost": -1.0,
                f"item|{ALIEN_POWER_MATRIX_CLASS}": -1.0,
            }
            self.add_lp_column(
                coeffs,
                type_="alien_power_matrix",
                name="fuel",
            )
            self.config.lp_equalities["alien_power_matrix_cost"] = -self._data.constants.TOTAL_ALIEN_POWER_MATRIX_COST

        # Configured extra costs
        for extra_cost, cost_variable, cost_coeff in [
            ("transport_power_cost", "power_consumption", 1.0),
            ("drone_battery_cost", f"item|{BATTERY_CLASS}", -1.0),
        ]:
            coeffs = {
                extra_cost: -1.0,
                cost_variable: cost_coeff,
            }
            self.add_lp_column(
                coeffs,
                type_="extra_cost",
                name=extra_cost,
            )
            self.config.lp_equalities[extra_cost] = 0.0

        # Somersloop usage
        coeffs = {
            "somersloop_usage": -1.0,
            "somersloop": -1.0,
        }
        self.add_lp_column(
            coeffs,
            type_="somersloop",
            name="usage",
        )
        self.config.lp_equalities["somersloop_usage"] = 0.0
        self.config.lp_lower_bounds["somersloop"] = -self._data.constants.NUM_SOMERSLOOPS_AVAILABLE_FOR_PRODUCTION

        self.result.lp_variables = get_all_variables(
            self.config.lp_columns, 
            self.config.lp_equalities, 
            self.config.lp_lower_bounds)

        ### Pruning unreachable items ###


        reachable_items: set[str] = set()
        while True:
            any_added = False
            for column_id, column in self.config.lp_columns.items():
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
            set(v for v in self.result.lp_variables if v.startswith("item|")) - reachable_items
        )

        self._dbug.debug_dump("Unreachable items to be pruned", unreachable_items)

        columns_to_prune: list[str] = []
        for column_id, column in self.config.lp_columns.items():
            for variable, coeff in column.coeffs.items():
                if variable in unreachable_items and coeff < 0:
                    columns_to_prune.append(column_id)
                    break
        for column_id in columns_to_prune:
            del self.config.lp_columns[column_id]
        for item_var in unreachable_items:
            if item_var in self.config.lp_equalities:
                del self.config.lp_equalities[item_var]


        self._dbug.debug_dump("LP columns (after pruning)", self.config.lp_columns)
        self._dbug.debug_dump("LP equalities (after pruning)", self.config.lp_equalities)
        self._dbug.debug_dump("LP lower bounds (after pruning)", self.config.lp_lower_bounds)

        lp_variables = get_all_variables(
            self.config.lp_columns, self.config.lp_equalities, self.config.lp_lower_bounds
        )
        self.result.lp_variables = lp_variables

        self._dbug.debug_dump("LP variables (after pruning)", lp_variables)


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
        
        self.result.extra_cost_order = extra_cost_order
        self.result.objective_order = objective_order
        self.result.resource_subtype_order = resource_subtype_order


        def column_order_key(
                arg: tuple[str, LPColumn]
                ):
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

        self.result.sorted_columns = sorted(self.config.lp_columns.items(), key=column_order_key)
        self.result.lp_variable_indices = to_index_map(lp_variables)

        _lp_c = np.zeros(len(self.config.lp_columns), dtype=np.float64)
        self.config.lp_integrality = np.zeros(len(self.config.lp_columns), dtype=np.int64)
        _lp_A = np.zeros((len(lp_variables), len(self.config.lp_columns)), dtype=np.float64)
        _lp_b_l = np.zeros(len(lp_variables), dtype=np.float64)
        _lp_b_u = np.zeros(len(lp_variables), dtype=np.float64)
    
        # copy lp bounds to result
        self.result.lp_lower_bounds = self.config.lp_lower_bounds
        self.config.lp_c = _lp_c
        self.result.lp_b_l = _lp_b_l
        self.result.lp_b_u = _lp_b_u
        self.result.lp_A = _lp_A

        for column_index, (column_id, column) in enumerate(self.result.sorted_columns):
            if column.objective_weight is not None:
                _lp_c[column_index] = column.objective_weight
            if column.requires_integrality:
                self.config.lp_integrality[column_index] = 1
            for variable, coeff in column.coeffs.items():
                _lp_A[self.result.lp_variable_indices[variable], column_index] = coeff

        for variable, rhs in self.config.lp_equalities.items():
            _lp_b_l[self.result.lp_variable_indices[variable]] = rhs
            _lp_b_u[self.result.lp_variable_indices[variable]] = rhs

        for variable, rhs in self.config.lp_lower_bounds.items():
            _lp_b_l[self.result.lp_variable_indices[variable]] = rhs
            _lp_b_u[self.result.lp_variable_indices[variable]] = np.inf

        self.config.lp_constraints = scipy.optimize.LinearConstraint(
            _lp_A, 
            cast(Any, _lp_b_l), 
            cast(Any, _lp_b_u))  



    def run(self) -> dict[str, Any]:
        print("LP running")

        _lp_c = self.config.lp_c


        self.result.lp_result = scipy.optimize.milp(
            -cast(np.float64, _lp_c),
            integrality=self.config.lp_integrality,
            constraints=self.config.lp_constraints,
        )

        if self.result.lp_result.status != 0:
            
            return {
                'status' : 502,
                'result' : self.result
            }
            


        return {
            'status' : 200,
            'result' : self.result
        }
    
    
    def add_objective_column(
            self,
            objective: str, objective_weight: float, hard_limit: HardLimit | None = None
        ):
        coeffs = {
            objective: -1.0,
        }
        if hard_limit is not None:
            coeffs[hard_limit.name] = hard_limit.weight
            self.config.lp_lower_bounds[hard_limit.name] = hard_limit.lower_bound
        self.add_lp_column(
            coeffs,
            type_="objective",
            name=objective,
            objective_weight=objective_weight,
        )
        self.config.lp_equalities[objective] = 0.0

    def add_meta_coeffs(self, column_id: str, column: LPColumn):
        to_add: defaultdict[str, float] = defaultdict(float)
        for variable, coeff in column.coeffs.items():
            if variable.startswith("item|") and coeff > 0:
                item_class = variable[5:]
                if item_class not in self._data.data.items:
                    print(f"WARNING: item not found in items dict: {item_class}")
                    continue

                item = self._data.data.items[item_class]
                form = item.form
                conveyance_limit = self._data.get_form_conveyance_limit(form)
                conveyance = coeff / conveyance_limit

                # Avoid incurring transport costs for Water Extractors,
                # as they would otherwise dominate the cost.
                # Basically we're assuming other stuff is brought to the water.
                if column.type_ == "miner" and column.resource_subtype != "extractor":
                    to_add["transport_power_cost"] += self._args.transport_power_cost * conveyance
                    to_add["drone_battery_cost"] += self._args.drone_battery_cost * conveyance

                if form == "RF_SOLID":
                    to_add["conveyors"] += conveyance
                else:
                    to_add["pipelines"] += conveyance

        for variable, coeff in to_add.items():
            if coeff != 0.0:
                column.coeffs[variable] = column.coeffs.get(variable, 0.0) + coeff

    def add_geothermal_generator_columns(self, resource: Resource):
        resource_id = resource.resource_id
        resource_var = f"resource|{resource_id}"

        power_production = (
            cast(GeothermalGenerator, self._data.data.geothermal_generator).mean_variable_power_production 
            * resource.multiplier
            * self._data.constants.POWER_PRODUCTION_MULTIPLIER
        )
        coeffs = {
            "machines": 1,
            "power_production": power_production,
            resource_var: -1,
        }

        self.add_lp_column(
            coeffs,
            type_="generator",
            name=resource_id,
            display_name=self._data.get_item_display_name(GEOTHERMAL_CLASS),
            machine_name=cast(GeothermalGenerator, self._data.data.geothermal_generator).display_name, 
            resource_subtype=resource.subtype,
            requires_integrality=True,
        )

        resource_multiplier = self.config.resource_multipliers.get(
            resource.item_class,
            self.config.resource_multipliers.get("All", self.config.resource_multipliers.get("all", 1.0)),
        )

        self.config.lp_lower_bounds[resource_var] = -resource.count * resource_multiplier

    def add_generator_columns(self, generator: PowerGenerator, fuel: Fuel):
        recipe = create_recipe_for_generator(generator, fuel, self._data.data)
        fuel_item = self._data.data.items[fuel.fuel_class]

        min_clock = generator.min_clock
        max_clock = self._data.get_max_recipe_clock(generator, recipe)
        clock_choices = clamp_clock_choices(self.config.lp_clocks.GENERATOR_CLOCKS, min_clock, max_clock)

        for clock in clock_choices:
            power_production = (
                get_power_production(generator, clock=clock) * self._data.constants.POWER_PRODUCTION_MULTIPLIER
            )
            coeffs = {
                "machines": 1,
                "power_production": power_production,
            }

            recipe_coeffs = get_recipe_coeffs(recipe, clock=clock)
            for item_var, coeff in recipe_coeffs.items():
                coeffs[item_var] = coeff
                self.config.lp_equalities[item_var] = 0.0

            self.add_lp_column(
                coeffs,
                type_="generator",
                name=fuel_item.class_name,
                display_name=fuel_item.display_name,
                machine_name=generator.display_name,
                clock=clock,
            )

    def add_sink_column(self, item: Item):
        item_class = item.class_name
        item_var = f"item|{item_class}"
        points = item.points

        if not (item.form == "RF_SOLID" and points > 0):
            if self._args.allow_waste:
                self.add_lp_column(
                    {item_var: -1},
                    type_="waste",
                    name=item_class,
                    display_name=item.display_name,
                )
            return

        coeffs = {
            "machines": 1 / self._data.constants.CONVEYOR_BELT_LIMIT,
            "power_consumption": self._data.constants.SINK_POWER_CONSUMPTION / self._data.constants.CONVEYOR_BELT_LIMIT,
            item_var: -1,
            "points": points,
        }

        self.add_lp_column(
            coeffs,
            type_="sink",
            name=item_class,
            display_name=item.display_name,
        )

        self.config.lp_equalities[item_var] = 0.0

    def add_manufacturer_columns(self, recipe: Recipe):
        manufacturer_class = recipe.manufacturer
        manufacturer = self._data.data.manufacturers[manufacturer_class]

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
            throughput_mult = base_output_mult * self._args.building_multiplier
            power_mult = base_power_mult * self._args.building_multiplier

            min_clock = manufacturer.min_clock
            max_clock = self._data.get_max_recipe_clock(
                manufacturer, recipe, throughput_multiplier=throughput_mult  # changed name
            )
            configured_clocks = self.config.lp_clocks.MANUFACTURER_CLOCKS if somersloops is None else self.config.lp_clocks.SOMERSLOOP_CLOCKS
            clock_choices = clamp_clock_choices(configured_clocks, min_clock, max_clock)

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
                    self.config.lp_equalities[item_var] = 0.0

                self.add_lp_column(
                    coeffs,
                    type_="manufacturer",
                    name=recipe.class_name,
                    display_name=recipe.display_name,
                    machine_name=manufacturer.display_name,
                    clock=clock,
                    somersloops=somersloops,
                    requires_integrality=requires_integrality,
                )

    def add_miner_columns(self, resource: Resource):
        resource_id = resource.resource_id
        item_class = resource.item_class
        item = self._data.data.items[item_class]

        miner = self._data.get_miner_for_resource(resource)
        

        extraction_rate = miner.extraction_rate_base * resource.multiplier
        min_clock = miner.min_clock
        max_clock = self._data.get_max_extraction_clock(miner, resource, extraction_rate)
        configured_clocks = self.config.lp_clocks.MANUFACTURER_CLOCKS if resource.is_unlimited else self.config.lp_clocks.MINER_CLOCKS


        clock_choices = clamp_clock_choices(configured_clocks, min_clock, max_clock)

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

            self.add_lp_column(
                coeffs,
                type_="miner",
                name=resource_id,
                display_name=item.display_name,
                machine_name=miner.display_name,
                resource_subtype=resource.subtype,
                clock=clock,
            )

        if not resource.is_unlimited:
            resource_multiplier = self.config.resource_multipliers.get(
                resource.item_class,
                self.config.resource_multipliers.get("All", self.config.resource_multipliers.get("all", 1.0)),
            )
            self.config.lp_lower_bounds[resource_var] = -resource.count * resource_multiplier

        self.config.lp_equalities[item_var] = 0.0

    def add_lp_column(self,
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
        assert column_id not in self.config.lp_columns, f"duplicate {column_id=}"
        self.config.lp_columns[column_id] = LPColumn(
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

