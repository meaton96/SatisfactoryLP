import json
from pathlib import Path

DOCS_IN = Path("Docs.json")
DOCS_OUT = Path("Docs.patched.json")

def ing(*pairs):
    # Build UE-style tuple string for mIngredients/mProduct
    parts = []
    for item_class_path, amount in pairs:
        parts.append(f'(ItemClass="{item_class_path}",Amount={amount})')
    return "(" + ",".join(parts) + ")"

# ItemClass paths taken from your Docs.json so nothing breaks:
ROCKET_FUEL = "/Script/Engine.BlueprintGeneratedClass'/Game/FactoryGame/Resource/Parts/RocketFuel/Desc_RocketFuel.Desc_RocketFuel_C'"
POWER_SHARD = "/Script/Engine.BlueprintGeneratedClass'/Game/FactoryGame/Resource/Environment/Crystal/Desc_CrystalShard.Desc_CrystalShard_C'"
IONIZED_FUEL = "/Script/Engine.BlueprintGeneratedClass'/Game/FactoryGame/Resource/Parts/IonizedFuel/Desc_IonizedFuel.Desc_IonizedFuel_C'"
COMPACTED_COAL = "/Script/Engine.BlueprintGeneratedClass'/Game/FactoryGame/Resource/Parts/CompactedCoal/Desc_CompactedCoal.Desc_CompactedCoal_C'"

DARK_MATTER_CRYSTAL = "/Script/Engine.BlueprintGeneratedClass'/Game/FactoryGame/Resource/Parts/DarkMatter/Desc_DarkMatter.Desc_DarkMatter_C'"
DARK_ENERGY = "/Script/Engine.BlueprintGeneratedClass'/Game/FactoryGame/Resource/Parts/DarkEnergy/Desc_DarkEnergy.Desc_DarkEnergy_C'"

PLUTONIUM_WASTE = "/Script/Engine.BlueprintGeneratedClass'/Game/FactoryGame/Resource/Parts/NuclearWaste/Desc_PlutoniumWaste.Desc_PlutoniumWaste_C'"
SINGULARITY_CELL = "/Script/Engine.BlueprintGeneratedClass'/Game/FactoryGame/Resource/Parts/SingularityCell/Desc_SingularityCell.Desc_SingularityCell_C'"
FICSONIUM = "/Script/Engine.BlueprintGeneratedClass'/Game/FactoryGame/Resource/Parts/Ficsonium/Desc_Ficsonium.Desc_Ficsonium_C'"

FICSONIUM_ROD = "/Script/Engine.BlueprintGeneratedClass'/Game/FactoryGame/Resource/Parts/FicsoniumFuelRod/Desc_FicsoniumFuelRod.Desc_FicsoniumFuelRod_C'"
E_CONTROL_ROD = "/Script/Engine.BlueprintGeneratedClass'/Game/FactoryGame/Resource/Parts/ElectromagneticControlRod/Desc_ElectromagneticControlRod.Desc_ElectromagneticControlRod_C'"
FICSITE_TRIGON = "/Script/Engine.BlueprintGeneratedClass'/Game/FactoryGame/Resource/Parts/FicsiteMesh/Desc_FicsiteMesh.Desc_FicsiteMesh_C'"
QUANTUM_ENERGY = "/Script/Engine.BlueprintGeneratedClass'/Game/FactoryGame/Resource/Parts/QuantumEnergy/Desc_QuantumEnergy.Desc_QuantumEnergy_C'"

def patch_recipe(entry):
    cn = entry.get("ClassName")
    if cn == "Recipe_IonizedFuel_C":
        entry["mIngredients"] = ing((ROCKET_FUEL, 50), (POWER_SHARD, 1))
        entry["mProduct"] = ing((IONIZED_FUEL, 50), (COMPACTED_COAL, 2))
        entry["mManufactoringDuration"] = "30.000000"
    elif cn == "Recipe_Alternate_IonizedFuel_Dark_C":
        entry["mIngredients"] = ing((ROCKET_FUEL, 25), (DARK_MATTER_CRYSTAL, 2))
        entry["mProduct"] = ing((IONIZED_FUEL, 20), (COMPACTED_COAL, 2))
        entry["mManufactoringDuration"] = "6.000000"
    elif cn == "Recipe_Ficsonium_C":
        entry["mIngredients"] = ing((PLUTONIUM_WASTE, 5), (SINGULARITY_CELL, 1), (DARK_ENERGY, 20))
        entry["mProduct"] = ing((FICSONIUM, 5))
        entry["mManufactoringDuration"] = "30.000000"
    elif cn == "Recipe_FicsoniumFuelRod_C":
        entry["mIngredients"] = ing((FICSONIUM, 10), (E_CONTROL_ROD, 10), (FICSITE_TRIGON, 40), (QUANTUM_ENERGY, 20))
        entry["mProduct"] = ing((FICSONIUM_ROD, 5), (DARK_ENERGY, 20))
        entry["mManufactoringDuration"] = "60.000000"

def walk_and_patch(docs):
    for fg in docs:
        native = fg.get("NativeClass", "")
        classes = fg.get("Classes", [])
        # Only recipes matter here
        if "FGRecipe" in native or any(c.get("ClassName","").startswith("Recipe_") for c in classes):
            for entry in classes:
                patch_recipe(entry)

def main():
    # Read UTF-16 like your solver does
    with open(DOCS_IN, "r", encoding="utf-16") as f:
        docs = json.load(f)

    walk_and_patch(docs)

    # Write back as UTF-16 for drop-in compatibility
    with open(DOCS_OUT, "w", encoding="utf-16") as f:
        json.dump(docs, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
