from __future__ import annotations
import json
from .config import *
from typing import Any, Iterable
import re
from .debug import Debugger
from .models import *
from .utils import parse_paren_list, find_item_amounts
from src.utils import *
from dataclasses import dataclass

### Load json ###




class DataParser:

    def __init__(self):
        self.docs_raw: Any
        self.args: Any
        
        self.dbug: Debugger
        self.data: Data
        self.constants: Consts

    

    def init(self, args, debug):
        self.args = args
        self.dbug = debug
        self.data = Data()
        self.constants = Consts()
        with open(DOCS_PATH, "r", encoding="utf-16") as f:
            self.docs_raw = json.load(f)
        self.parse()

        

    def get_miner_for_resource(self,resource: Resource) -> Miner:
        item_class = resource.item_class
        item = self.data.items[item_class]
        candidates: list[Miner] = []
        for miner in self.data.miners.values():
            if (
                miner.uses_resource_wells == resource.is_resource_well
                and miner.check_allowed_resource(item_class, item.form)
            ):
                candidates.append(miner)
        assert candidates, f"could not find miner for {item_class}"
        assert len(candidates) == 1, f"more than one miner for {item_class}: {candidates}"
        return candidates[0]


    def get_form_conveyance_limit(self,form: str) -> float:

        if form == "RF_SOLID":
            return self.constants.CONVEYOR_BELT_LIMIT
        elif form == "RF_LIQUID" or form == "RF_GAS":
            return self.constants.PIPELINE_LIMIT
        else:
            assert False


    def get_conveyance_limit_clock(self,item: Item, rate: float) -> Fraction:
        conveyance_limit = self.get_form_conveyance_limit(item.form)
        a = float_to_clock(conveyance_limit / rate)
        print(f"get conveyance limit: {item.display_name} : {item.form} : {conveyance_limit} : {a}")
        return a


    def get_max_extraction_clock(self,
        miner: Miner, resource: Resource, extraction_rate: float
    ) -> Fraction:
        max_clock = miner.max_clock

        # Assume individual Resource Well Extractors can never exceed conveyance limit
        if resource.is_resource_well:
            return max_clock

        item = self.data.items[resource.item_class]
        return min(max_clock, self.get_conveyance_limit_clock(item, extraction_rate))


    def get_max_recipe_clock(self,
        machine: Machine, recipe: Recipe, throughput_multiplier: float = 1.0
    ) -> Fraction:
        max_clock = machine.max_clock

        for item_class, input_rate in recipe.inputs:
            max_clock = min(
                max_clock,
                self.get_conveyance_limit_clock(self.data.items[item_class], input_rate * throughput_multiplier),
            )

        for item_class, output_rate in recipe.outputs:
            max_clock = min(
                max_clock,
                self.get_conveyance_limit_clock(self.data.items[item_class], output_rate * throughput_multiplier),
            )

        return max_clock



    def clamp_clock_choices(self,
        configured_clocks: list[Fraction], min_clock: Fraction, max_clock: Fraction
    ) -> list[Fraction]:
        

        assert min_clock < max_clock
        return sorted(
            {min(max_clock, max(min_clock, clock)) for clock in configured_clocks}
        )


    


    def get_item_display_name(self,item_class: str) -> str:
        if item_class in self.data.items:
            return self.data.items[item_class].display_name
        else:
            return ADDITIONAL_DISPLAY_NAMES[item_class]
        
    def parse(self):
        ### Initial parsing ###
        class_name_to_entry: dict[str, dict[str, Any]] = {}
        native_class_to_class_entries: dict[str, list[dict[str, Any]]] = {}

        NATIVE_CLASS_REGEX = re.compile(r"/Script/CoreUObject.Class'/Script/FactoryGame.(\w+)'")


        def parse_and_add_fg_entry(fg_entry: dict[str, Any], merge: bool = False):
            m = NATIVE_CLASS_REGEX.fullmatch(fg_entry["NativeClass"])
            assert m is not None, fg_entry["NativeClass"]
            native_class = m.group(1)

            class_entries: list[dict[str, Any]] = []
            for class_entry in fg_entry["Classes"]:
                class_name = class_entry["ClassName"]
                if not merge and class_name in class_name_to_entry:
                    print(f"WARNING: ignoring duplicate class {class_name}")
                else:
                    class_name_to_entry[class_name] = class_entry
                    class_entries.append(class_entry)
            native_class_to_class_entries[native_class] = class_entries

        for fg_entry in self.docs_raw:
            parse_and_add_fg_entry(fg_entry)

        def parse_modded_docs():
            for p in self.args.extra_docs:
                with open(p, "r", encoding="utf-16") as f:
                    extra_raw = json.load(f)
                for fg_entry in extra_raw:
                    parse_and_add_fg_entry(fg_entry, merge=True)  

        self.constants.CONVEYOR_BELT_LIMIT = 0.5 * float(class_name_to_entry[CONVEYOR_BELT_CLASS]["mSpeed"])
        self.constants.PIPELINE_LIMIT = 60000.0 * float(class_name_to_entry[PIPELINE_CLASS]["mFlowLimit"])
        self.constants.SINK_POWER_CONSUMPTION = float(class_name_to_entry[SINK_CLASS]["mPowerConsumption"])


        self.dbug.debug_dump(
            "Misc constants",
            f"""
        {self.constants.CONVEYOR_BELT_LIMIT=}
        {self.constants.PIPELINE_LIMIT=}
        {self.constants.SINK_POWER_CONSUMPTION=}
        """.strip(),
        )


        ALIEN_POWER_AUGMENTER_STATIC_POWER = float(
            class_name_to_entry[ALIEN_POWER_AUGMENTER_CLASS]["mBasePowerProduction"]
        )
        ALIEN_POWER_AUGMENTER_BASE_CIRCUIT_BOOST = float(
            class_name_to_entry[ALIEN_POWER_AUGMENTER_CLASS]["mBaseBoostPercentage"]
        )
        ALIEN_POWER_AUGMENTER_FUELED_CIRCUIT_BOOST = (
            ALIEN_POWER_AUGMENTER_BASE_CIRCUIT_BOOST
            + float(class_name_to_entry[ALIEN_POWER_MATRIX_CLASS]["mBoostPercentage"])
        )
        ALIEN_POWER_AUGMENTER_FUEL_INPUT_RATE = 60.0 / float(
            class_name_to_entry[ALIEN_POWER_MATRIX_CLASS]["mBoostDuration"]
        )


        self.dbug.debug_dump(
            "Alien Power Augmenter constants",
            f"""
        {ALIEN_POWER_AUGMENTER_STATIC_POWER=}
        {ALIEN_POWER_AUGMENTER_BASE_CIRCUIT_BOOST=}
        {ALIEN_POWER_AUGMENTER_FUELED_CIRCUIT_BOOST=}
        {ALIEN_POWER_AUGMENTER_FUEL_INPUT_RATE=}
        """.strip(),
        )




        ### Miners ###


        def parse_miner(entry: dict[str, Any]) -> Miner:
            # Resource well extractors are aggregated under the pressurizer
            if entry["ClassName"] == RESOURCE_WELL_PRESSURIZER_CLASS:
                extractor = class_name_to_entry[RESOURCE_WELL_EXTRACTOR_CLASS]
                uses_resource_wells = True
            else:
                extractor = entry
                uses_resource_wells = False

            assert entry["mCanChangePotential"] == "True"
            assert str_to_clock(entry["mMaxPotential"]) == MACHINE_BASE_CLOCK

            assert entry["mCanChangeProductionBoost"] == "False"

            # This will be multiplied by the purity when applied to a node
            # (or, for resource wells, the sum of satellite purities).
            extraction_rate_base = (
                60.0
                / float(extractor["mExtractCycleTime"])
                * float(extractor["mItemsPerCycle"])
            )

            allowed_resource_forms = parse_paren_list(extractor["mAllowedResourceForms"])
            assert allowed_resource_forms is not None

            return Miner(
                class_name=entry["ClassName"],
                display_name=entry["mDisplayName"],
                power_consumption=float(entry["mPowerConsumption"]),
                power_consumption_exponent=float(entry["mPowerConsumptionExponent"]),
                min_clock=str_to_clock(entry["mMinPotential"]),
                max_clock=float_to_clock(MACHINE_MAX_CLOCK),
                is_variable_power=False,
                extraction_rate_base=extraction_rate_base,
                uses_resource_wells=uses_resource_wells,
                allowed_resource_forms=allowed_resource_forms,
                only_allow_certain_resources=(
                    extractor["mOnlyAllowCertainResources"] == "True"
                ),
                allowed_resources=parse_class_list(extractor["mAllowedResources"], self.constants),
            )


        

        for class_name in ALL_MINER_CLASSES:
            self.data.miners[class_name] = parse_miner(class_name_to_entry[class_name])

        self.dbug.debug_dump("Parsed miners", self.data.miners)


        ### Manufacturers ###


        def parse_manufacturer(entry: dict[str, Any], is_variable_power: bool) -> Manufacturer:
            class_name = entry["ClassName"]

            assert entry["mCanChangePotential"] == "True"
            assert str_to_clock(entry["mMaxPotential"]) == MACHINE_BASE_CLOCK

            can_change_production_boost = entry["mCanChangeProductionBoost"] == "True"
            production_shard_slot_size = int(entry["mProductionShardSlotSize"])

            # Smelter has "mProductionShardSlotSize": "0" when it should be 1
            if class_name == "Build_SmelterMk1_C":
                assert can_change_production_boost
                if production_shard_slot_size == 0:
                    production_shard_slot_size = 1

            return Manufacturer(
                class_name=class_name,
                display_name=entry["mDisplayName"],
                power_consumption=float(entry["mPowerConsumption"]),
                power_consumption_exponent=float(entry["mPowerConsumptionExponent"]),
                min_clock=str_to_clock(entry["mMinPotential"]),
                max_clock=float_to_clock(MACHINE_MAX_CLOCK),
                is_variable_power=is_variable_power,
                can_change_production_boost=(entry["mCanChangeProductionBoost"] == "True"),
                base_production_boost=float(entry["mBaseProductionBoost"]),
                production_shard_slot_size=production_shard_slot_size,
                production_shard_boost_multiplier=float(
                    entry["mProductionShardBoostMultiplier"]
                ),
                production_boost_power_consumption_exponent=float(
                    entry["mProductionBoostPowerConsumptionExponent"]
                ),
            )


        

        for entry in native_class_to_class_entries["FGBuildableManufacturer"]:
            manufacturer = parse_manufacturer(entry, is_variable_power=False)
            self.data.manufacturers[manufacturer.class_name] = manufacturer

        for entry in native_class_to_class_entries["FGBuildableManufacturerVariablePower"]:
            manufacturer = parse_manufacturer(entry, is_variable_power=True)
            self.data.manufacturers[manufacturer.class_name] = manufacturer

        self.dbug.debug_dump("Parsed manufacturers", self.data.manufacturers)


        ### Recipes ###


        def parse_recipe(entry: dict[str, Any]) -> Recipe | None:
            produced_in = parse_class_list(entry["mProducedIn"], self.constants) or []
            recipe_manufacturer = None

            for manufacturer in produced_in:
                if manufacturer in self.data.manufacturers:
                    recipe_manufacturer = manufacturer
                    break

            if recipe_manufacturer is None:
                # check that recipe is not automatable for known reasons
                assert (
                    not produced_in
                    or "BP_WorkshopComponent_C" in produced_in
                    or "BP_BuildGun_C" in produced_in
                    or "FGBuildGun" in produced_in
                ), f"{entry["mDisplayName"]} {produced_in}"
                return None

            recipe_rate = 60.0 / float(entry["mManufactoringDuration"])

            def item_rates(key: str):
                return [
                    (item, recipe_rate * amount)
                    for (item, amount) in find_item_amounts(entry[key], self.constants)
                ]

            vpc_constant = float(entry["mVariablePowerConsumptionConstant"])
            vpc_factor = float(entry["mVariablePowerConsumptionFactor"])
            # Assuming the mean is exactly halfway for all of the variable power machine types.
            # This appears to be accurate but it's hard to confirm exactly.
            mean_variable_power_consumption = vpc_constant + 0.5 * vpc_factor

            return Recipe(
                class_name=entry["ClassName"],
                display_name=entry["mDisplayName"],
                manufacturer=recipe_manufacturer,
                inputs=item_rates("mIngredients"),
                outputs=item_rates("mProduct"),
                mean_variable_power_consumption=mean_variable_power_consumption,
            )


        

        for entry in native_class_to_class_entries["FGRecipe"]:
            recipe = parse_recipe(entry)
            if recipe is not None:
                self.data.recipes[recipe.class_name] = recipe


        self.dbug.debug_dump("Parsed recipes", self.data.recipes)


        ### Items ###


        def parse_item(entry: dict[str, Any]) -> Item:
            return Item(
                class_name=entry["ClassName"],
                display_name=entry["mDisplayName"],
                form=entry["mForm"],
                points=int(entry["mResourceSinkPoints"]),
                stack_size=STACK_SIZES[entry["mStackSize"]],
                energy=float(entry["mEnergyValue"]),
            )


        


        for native_class in ALL_ITEM_NATIVE_CLASSES:
            for entry in native_class_to_class_entries[native_class]:
                item = parse_item(entry)
                self.data.items[item.class_name] = item

        self.dbug.debug_dump("Parsed items", self.data.items)


        ### Generators ###


        def parse_fuel(entry: dict[str, Any]) -> Fuel:
            byproduct_amount = entry["mByproductAmount"]
            return Fuel(
                fuel_class=entry["mFuelClass"],
                supplemental_resource_class=entry["mSupplementalResourceClass"] or None,
                byproduct=entry["mByproduct"] or None,
                byproduct_amount=int(byproduct_amount) if byproduct_amount else 0,
            )


        def parse_generator(entry: dict[str, Any]) -> PowerGenerator:
            fuels = [parse_fuel(fuel) for fuel in entry["mFuel"]]

            assert entry["mCanChangePotential"] == "True"
            assert str_to_clock(entry["mMaxPotential"]) == MACHINE_BASE_CLOCK

            return PowerGenerator(
                class_name=entry["ClassName"],
                display_name=entry["mDisplayName"],
                fuels=fuels,
                power_production=float(entry["mPowerProduction"]),
                min_clock=str_to_clock(entry["mMinPotential"]),
                max_clock=float_to_clock(MACHINE_MAX_CLOCK),
                requires_supplemental=(entry["mRequiresSupplementalResource"] == "True"),
                supplemental_to_power_ratio=float(entry["mSupplementalToPowerRatio"]),
            )


        def parse_geothermal_generator(entry: dict[str, Any]) -> GeothermalGenerator:
            # "mVariablePowerProductionConstant": "0.000000" should be 100 MW, hardcode it here
            vpp_constant = 100.0
            vpp_factor = float(entry["mVariablePowerProductionFactor"])
            # Assuming the mean power production is exactly halfway.
            mean_variable_power_production = vpp_constant + 0.5 * vpp_factor

            assert entry["mCanChangePotential"] == "False"

            return GeothermalGenerator(
                class_name=entry["ClassName"],
                display_name=entry["mDisplayName"],
                min_clock=float_to_clock(MACHINE_BASE_CLOCK),
                max_clock=float_to_clock(MACHINE_BASE_CLOCK),
                mean_variable_power_production=mean_variable_power_production,
            )


        

        for native_class in ALL_GENERATOR_NATIVE_CLASSES:
            for entry in native_class_to_class_entries[native_class]:
                generator = parse_generator(entry)
                self.data.generators[generator.class_name] = generator

        # geothermal generator (special case)
        self.data.geothermal_generator = parse_geothermal_generator(
            class_name_to_entry[GEOTHERMAL_GENERATOR_CLASS]
        )

        self.dbug.debug_dump("Parsed generators", self.data.generators)
        self.dbug.debug_dump("Parsed geothermal generator", self.data.geothermal_generator)


        ### Map info ###


        with open(MAP_INFO_PATH, "r") as f:
            map_info_raw = json.load(f)

        

        for tab in map_info_raw["options"]:
            if "tabId" in tab:
                self.data.map_info[tab["tabId"]] = tab["options"]


        ### Resources ###





        


        # Persistent_Level:PersistentLevel.BP_FrackingCore6_UAID_40B076DF2F79D3DF01_1961476789
        # becomes #6. We can strip out the UAID as long as it's unique for each item type.
        FRACKING_CORE_REGEX = re.compile(
            r"Persistent_Level:PersistentLevel\.BP_FrackingCore_?(\d+)(_UAID_\w+)?"
        )


        def parse_fracking_core_name(s: str) -> str:
            m = FRACKING_CORE_REGEX.fullmatch(s)
            assert m is not None, s
            return "#" + m.group(1)


        def parse_and_add_resources(map_resource: dict[str, Any]):
            if "type" not in map_resource:
                return

            item_class = map_resource["type"]
            if item_class in RESOURCE_MAPPINGS:
                item_class = RESOURCE_MAPPINGS[item_class]

            if item_class == GEYSER_CLASS:
                output = self.data.geysers
            else:
                output =  self.data.resources
                assert item_class in self.data.items, f"map has unknown resource: {item_class}"

            for node_purity in map_resource["options"]:
                purity = node_purity["purity"]
                nodes = node_purity["markers"]
                if not nodes:
                    continue
                sample_node = nodes[0]
                if "core" in sample_node:
                    # resource well satellite nodes, map to cores and sum the purity multipliers
                    for node in nodes:
                        subtype = parse_fracking_core_name(node["core"])
                        resource_id = f"{item_class}|{subtype}"
                        if resource_id not in output:
                            output[resource_id] = Resource(
                                resource_id=resource_id,
                                item_class=item_class,
                                subtype=subtype,
                                multiplier=0.0,
                                is_unlimited=False,
                                count=1,
                                is_resource_well=True,
                                num_satellites=0,
                            )
                        output[resource_id].multiplier += PURITY_MULTIPLIERS[purity]
                        output[resource_id].num_satellites += 1
                else:
                    # normal nodes, add directly
                    subtype = purity  # individual nodes are indistinguishable
                    resource_id = f"{item_class}|{subtype}"
                    assert resource_id not in output
                    output[resource_id] = Resource(
                        resource_id=resource_id,
                        item_class=item_class,
                        subtype=subtype,
                        multiplier=PURITY_MULTIPLIERS[purity],
                        is_unlimited=False,
                        count=len(nodes),
                        is_resource_well=False,
                        num_satellites=0,
                    )


        for map_resource in self.data.map_info["resource_nodes"]:
            parse_and_add_resources(map_resource)

        for map_resource in self.data.map_info["resource_wells"]:
            parse_and_add_resources(map_resource)

        # Water from extractors is a special infinite resource
        self.data.resources[f"{WATER_CLASS}|extractor"] = Resource(
            resource_id=f"{WATER_CLASS}|extractor",
            item_class=WATER_CLASS,
            subtype="extractor",
            multiplier=1,
            is_unlimited=True,
            count=0,
            is_resource_well=False,
            num_satellites=0,
        )

        self.dbug.debug_dump("Parsed resources",  self.data.resources)
        self.dbug.debug_dump("Parsed geysers",  self.data.geysers)


        ### Somersloops ###


        def find_somersloops_map_layer(
            map_tab_artifacts: list[dict[str, list[dict[str, Any]]]]
        ):
            for unknown_level in map_tab_artifacts:
                for map_layer in unknown_level["options"]:
                    if map_layer["layerId"] == "somersloops":
                        return map_layer
            raise RuntimeError("failed to find somersloops map layer")


        def parse_num_somersloops_on_map(somersloops_map_layer: dict[str, Any]) -> int:
            count = 0
            for marker in somersloops_map_layer["markers"]:
                if marker["type"] == "somersloop":
                    count += 1
            return count


        PRODUCTION_AMPLIFIER_UNLOCK_SOMERSLOOP_COST = 1
        ALIEN_POWER_AUGMENTER_UNLOCK_SOMERSLOOP_COST = 1
        ALIEN_POWER_AUGMENTER_BUILD_SOMERSLOOP_COST = 10


        def get_num_somersloops_available() -> int:
            if self.args.num_somersloops_available is not None:
                return self.args.num_somersloops_available

            num_somersloops_on_map = parse_num_somersloops_on_map(
                find_somersloops_map_layer(self.data.map_info["artifacts"])
            )
            research_somersloop_cost = (
                PRODUCTION_AMPLIFIER_UNLOCK_SOMERSLOOP_COST
                if not self.args.disable_production_amplification
                else 0
            ) + (
                ALIEN_POWER_AUGMENTER_UNLOCK_SOMERSLOOP_COST
                if self.args.num_alien_power_augmenters > 0
                else 0
            )
            assert research_somersloop_cost <= num_somersloops_on_map
            return num_somersloops_on_map - research_somersloop_cost


        NUM_SOMERSLOOPS_AVAILABLE = get_num_somersloops_available()
        POWER_SOMERSLOOP_COST: int = (
            ALIEN_POWER_AUGMENTER_BUILD_SOMERSLOOP_COST * self.args.num_alien_power_augmenters
        )

        assert (
            POWER_SOMERSLOOP_COST <= NUM_SOMERSLOOPS_AVAILABLE
        ), f"{POWER_SOMERSLOOP_COST=} {NUM_SOMERSLOOPS_AVAILABLE=}"

        NUM_SOMERSLOOPS_AVAILABLE_FOR_PRODUCTION = (
            NUM_SOMERSLOOPS_AVAILABLE - POWER_SOMERSLOOP_COST
        )

        assert self.args.num_fueled_alien_power_augmenters <= self.args.num_alien_power_augmenters

        self.constants.ALIEN_POWER_AUGMENTER_TOTAL_STATIC_POWER = (
            ALIEN_POWER_AUGMENTER_STATIC_POWER * self.args.num_alien_power_augmenters
        )
        self.constants.ALIEN_POWER_AUGMENTER_TOTAL_CIRCUIT_BOOST = (
            ALIEN_POWER_AUGMENTER_BASE_CIRCUIT_BOOST
            * (self.args.num_alien_power_augmenters - self.args.num_fueled_alien_power_augmenters)
        ) + (
            ALIEN_POWER_AUGMENTER_FUELED_CIRCUIT_BOOST * self.args.num_fueled_alien_power_augmenters
        )
        self.constants.POWER_PRODUCTION_MULTIPLIER = 1.0 + self.constants.ALIEN_POWER_AUGMENTER_TOTAL_CIRCUIT_BOOST
        self.constants.TOTAL_ALIEN_POWER_MATRIX_COST = (
            ALIEN_POWER_AUGMENTER_FUEL_INPUT_RATE * self.args.num_fueled_alien_power_augmenters
        )

        self.dbug.debug_dump(
            "Somersloops",
            f"""
        {self.args.disable_production_amplification=}
        {self.args.num_alien_power_augmenters=}
        {self.args.num_fueled_alien_power_augmenters=}
        {NUM_SOMERSLOOPS_AVAILABLE=}
        {NUM_SOMERSLOOPS_AVAILABLE_FOR_PRODUCTION=}
        {self.constants.ALIEN_POWER_AUGMENTER_TOTAL_STATIC_POWER=}
        {self.constants.ALIEN_POWER_AUGMENTER_TOTAL_CIRCUIT_BOOST=}
        {self.constants.POWER_PRODUCTION_MULTIPLIER=}
        {self.constants.TOTAL_ALIEN_POWER_MATRIX_COST=}
        """.strip(),
        )


        ### Additional helpers ###


        












### Parsing helpers ###














### Misc constants ###




