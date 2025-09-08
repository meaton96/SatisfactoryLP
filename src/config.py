#!/usr/bin/env python
# pyright: basic

"""
This module contains constants used throughout the Satisfactory LP solver.
"""

# Common
STACK_SIZES = {
    "SS_HUGE": 500,
    "SS_BIG": 200,
    "SS_MEDIUM": 100,
    "SS_SMALL": 50,
    "SS_ONE": 1,
    "SS_FLUID": 50000,
}

# Clock speeds
INVERSE_CLOCK_GRANULARITY = 100 * 10000

# Logistics
CONVEYOR_BELT_CLASS = "Build_ConveyorBeltMk6_C"
PIPELINE_CLASS = "Build_PipelineMK2_C"

# Miners
MINER_CLASS = "Build_MinerMk3_C"
OIL_EXTRACTOR_CLASS = "Build_OilPump_C"
WATER_EXTRACTOR_CLASS = "Build_WaterPump_C"
RESOURCE_WELL_EXTRACTOR_CLASS = "Build_FrackingExtractor_C"
RESOURCE_WELL_PRESSURIZER_CLASS = "Build_FrackingSmasher_C"
ALL_MINER_CLASSES = (
    MINER_CLASS,
    OIL_EXTRACTOR_CLASS,
    WATER_EXTRACTOR_CLASS,
    RESOURCE_WELL_PRESSURIZER_CLASS,
)

# Sink
SINK_CLASS = "Build_ResourceSink_C"

# Items
ALL_ITEM_NATIVE_CLASSES = (
    "FGItemDescriptor",
    "FGItemDescriptorBiomass",
    "FGItemDescriptorNuclearFuel",
    "FGItemDescriptorPowerBoosterFuel",
    "FGResourceDescriptor",
    "FGEquipmentDescriptor",
    "FGConsumableDescriptor",
    "FGPowerShardDescriptor",
    "FGAmmoTypeProjectile",
    "FGAmmoTypeInstantHit",
    "FGAmmoTypeSpreadshot",
)

# Water
WATER_CLASS = "Desc_Water_C"

# Generators (excl. geothermal)
ALL_GENERATOR_NATIVE_CLASSES = (
    "FGBuildableGeneratorFuel",
    "FGBuildableGeneratorNuclear",
)

# Geothermal generator
GEOTHERMAL_GENERATOR_CLASS = "Build_GeneratorGeoThermal_C"
GEYSER_CLASS = "Desc_Geyser_C"
GEOTHERMAL_CLASS = "Desc_Geyser_C"

# Alien Power Augmenter
ALIEN_POWER_AUGMENTER_CLASS = "Build_AlienPowerBuilding_C"
ALIEN_POWER_MATRIX_CLASS = "Desc_AlienPowerFuel_C"

# Resource map
PURITY_MULTIPLIERS = {
    "impure": 0.5,
    "normal": 1.0,
    "pure": 2.0,
}
RESOURCE_MAPPINGS = {
    "Desc_LiquidOilWell_C": "Desc_LiquidOil_C",
}

# Miscellaneous
BATTERY_CLASS = "Desc_Battery_C"
ADDITIONAL_DISPLAY_NAMES = {
    GEYSER_CLASS: "Geyser",
}

# Debug
DEBUG_INFO_PATH = r"DebugInfo.txt"
PPRINT_WIDTH = 120

# Somersloops
PRODUCTION_AMPLIFIER_UNLOCK_SOMERSLOOP_COST = 1
ALIEN_POWER_AUGMENTER_UNLOCK_SOMERSLOOP_COST = 1
ALIEN_POWER_AUGMENTER_BUILD_SOMERSLOOP_COST = 10

# JSON files
DOCS_PATH = r"src/data/Docs.json"
MAP_INFO_PATH = r"src/data/MapInfo.json"

# Clock speeds
MACHINE_BASE_CLOCK = 1.0
MACHINE_MAX_CLOCK = 2.5
