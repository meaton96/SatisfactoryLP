import argparse

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--machine-penalty",
        type=float,
        default=1000.0,
        help="objective penalty per machine built",
    )
    parser.add_argument(
        "--extra-docs", 
        type=str, 
        action="append", 
        default=[],
        help="Additional Docs-like JSON files to overlay (same schema as Docs.json)"
    )
    parser.add_argument(
        "--conveyor-penalty",
        type=float,
        default=0.0,
        help="objective penalty per conveyor belt of machine input/output",
    )
    parser.add_argument(
        "--building-multiplier",
        type=float,
        default=1.0,
        help="global speed multiplier for ALL manufacturers (throughput and per-machine power scale linearly)",
    )
    parser.add_argument(
        "--pipeline-penalty",
        type=float,
        default=0.0,
        help="objective penalty per pipeline of machine input/output",
    )
    parser.add_argument(
        "--machine-limit",
        type=float,
        help="hard limit on number of machines built",
    )
    parser.add_argument(
        "--transport-power-cost",
        type=float,
        default=0.0,
        help="added power cost to simulate transport per conveyor/pipeline of mined resource",
    )
    parser.add_argument(
        "--drone-battery-cost",
        type=float,
        default=0.0,
        help="added battery cost to simulate drone transport per conveyor/pipeline of mined resource",
    )
    parser.add_argument(
        "--miner-clocks",
        type=str,
        default="2.5",
        help="clock choices for miners (excluding Water Extractors)",
    )
    parser.add_argument(
        "--manufacturer-clocks",
        type=str,
        default="0-2.5/0.05",
        help="clock choices for non-somerslooped manufacturers (plus Water Extractors)",
    )
    parser.add_argument(
        "--somersloop-clocks",
        type=str,
        default="2.5",
        help="clock choices for somerslooped manufacturers",
    )
    parser.add_argument(
        "--generator-clocks",
        type=str,
        default="2.5",
        help="clock choices for power generators",
    )
    parser.add_argument(
        "--num-alien-power-augmenters",
        type=int,
        default=0,
        help="number of Alien Power Augmenters to build",
    )
    parser.add_argument(
        "--num-fueled-alien-power-augmenters",
        type=float,
        default=0,
        help="number of Alien Power Augmenters to fuel with Alien Power Matrix",
    )
    parser.add_argument(
        "--disable-production-amplification",
        action="store_true",
        help="disable usage of somersloops in manufacturers",
    )
    parser.add_argument(
        "--resource-multipliers",
        type=str,
        default="",
        help="comma-separated list of item_class:multiplier to scale resource node availability; supports All:<x> as a default",
    )
    parser.add_argument(
        "--num-somersloops-available",
        type=int,
        help="override number of somersloops available for production and APAs",
    )
    parser.add_argument(
        "--disabled-recipes",
        type=str,
        default="",
        help="comma-separated list of recipe_class to disable",
    )
    parser.add_argument(
        "--infinite-power",
        action="store_true",
        help="allow free infinite power consumption",
    )
    parser.add_argument(
        "--allow-waste",
        action="store_true",
        help="allow accumulation of nuclear waste and other unsinkable items",
    )
    parser.add_argument(
        "--show-unused",
        action="store_true",
        help="show unused LP columns (coeff 0) in the optimization result",
    )
    parser.add_argument(
        "--dump-debug-info",
        action="store_true",
        help="dump debug info to DebugInfo.txt (items, recipes, LP matrix, etc.)",
    )
    parser.add_argument(
        "--xlsx-report",
        type=str,
        default="Report",
        help="path to xlsx report output (empty string to disable)",
    )
    parser.add_argument(
        "--xlsx-sheet-suffix",
        type=str,
        default="",
        help="suffix to add to xlsx sheet names",
    )
    return parser
