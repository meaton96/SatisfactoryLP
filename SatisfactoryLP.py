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
from src.lp_utils import *
from src.setup_lp import SetupLP
import sys


T = TypeVar("T")


arg_parser = get_parser()
args = arg_parser.parse_args()

### Debug ###

dbug = Debugger()

if (args.dump_debug_info):
    dbug.init()

data_parser = DataParser()
data_parser.init(args, dbug)

lp_setup = SetupLP()
lp_setup.configure(args, dbug, data_parser)
lp_setup.run_setup()
response = lp_setup.run()

if not response['status'] == 200:
    print("ERROR: LP did not terminate successfully")
    pprint(response['result'])
    sys.exit(1)


result_obj = response['result']

print("LP result:")
pprint(result_obj.lp_result)

lp_result = result_obj.lp_result
sorted_columns = result_obj.sorted_columns
objective_order = result_obj.objective_order
resource_subtype_order = result_obj.resource_subtype_order
extra_cost_order = result_obj.extra_cost_order
lp_variables = result_obj.lp_variables
lp_A = result_obj.lp_A
lp_b_l = result_obj.lp_b_l
lp_b_u = result_obj.lp_b_u
lp_variable_indices = result_obj.lp_variable_indices
lp_lower_bounds = result_obj.lp_lower_bounds 




# dbug.debug_dump("LP variables (before pruning)", lp_variables)













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





variable_breakdowns = {
    variable: create_empty_variable_breakdown(
        variable,
        data_parser,
        variable_type_order,
        resource_subtype_order
        ) for variable in lp_variables
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

#pprint(lp_variable_indices)

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

