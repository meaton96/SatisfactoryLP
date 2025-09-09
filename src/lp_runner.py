from typing import Any
from src.utils import *
from src.config import *
from src.models import  *
from src.debug import Debugger
from src.data_parser import DataParser
from src.lp_utils import *
from src.setup_lp import SetupLP
import csv
import io


class LPRunner:
    def __init__(self, args: Any, extra_docs_data: str | dict | None = None):
        self.args = args
        self.dbug = Debugger()
        if self.args.dump_debug_info:
            self.dbug.init()

        self.data_parser = DataParser()
        self.data_parser.init(self.args, self.dbug, extra_docs_data=extra_docs_data)

        self.lp_setup = SetupLP()
        self.lp_setup.configure(self.args, self.dbug, self.data_parser)

    def run(self):
        self.lp_setup.run_setup()
        response = self.lp_setup.run()

        if not response['status'] == 200:
            return {
                "status": "error",
                "message": "LP did not terminate successfully",
                "data": response['result']
            }

        result_obj = response['result']
        column_results, variable_breakdowns, lp_objective = self.process_results(result_obj)

        if getattr(self.args, 'output_fmt', None) == 'csv':
            csv_str = self.format_csv(column_results, result_obj)
            return {
                "status": "success",
                "data": csv_str
            }

        

        return {
            "status": "success",
            "data": {
                "lp_result": result_obj.lp_result,
                "column_results": column_results,
                "variable_breakdowns": variable_breakdowns,
                "lp_objective": lp_objective
            }
        }

    def format_csv(self, column_results, result_obj):
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "recipe_name", "display_name", "machine_name", "clock_multiplier", "num_machines", "direction", "item_class", "rate", "is_fluid", "belt_count", "pipe_count"
        ])
        BELT_RATE = 1200.0
        PIPE_RATE = 600.0
        for column_id, column, column_coeff in column_results:
            if column.type_ != "manufacturer":
                continue
            num_machines = column_coeff
            machine_name = column.machine_name or ""
            clock_mult = column.clock if column.clock is not None else "1.0"
            display_name = column.display_name or ""
            for item_class, rate in column.coeffs.items():
                if not item_class.startswith("item|"):
                    continue
                direction = "input" if rate < 0 else "output"
                abs_rate = abs(rate * num_machines)
                item_short = item_class[5:]
                # Determine fluid status
                is_fluid = "yes" if any(fluid in item_short for fluid in FLUID_CLASSES) else "no"
                # Belt/Pipe calculation
                if is_fluid == "yes":
                    belt_count = "NA"
                    pipe_count = f"{abs_rate / PIPE_RATE:.2f}"
                else:
                    belt_count = f"{abs_rate / BELT_RATE:.2f}"
                    pipe_count = "NA"
                writer.writerow([
                    column.display_name,
                    display_name,
                    machine_name,
                    clock_mult,
                    f"{num_machines:.2f}",
                    direction,
                    item_short,
                    f"{abs_rate:.2f}",
                    is_fluid,
                    belt_count,
                    pipe_count
                ])
        return output.getvalue()

    def process_results(self, result_obj):
        REPORT_EPSILON = 1e-7

        column_results: list[tuple[str, LPColumn, float]] = [
            (column_id, column, result_obj.lp_result.x[column_index])
            for column_index, (column_id, column) in enumerate(result_obj.sorted_columns)
        ]

        if not self.args.show_unused:
            column_results = list(filter(lambda x: abs(x[2]) > REPORT_EPSILON, column_results))

        variable_type_order = to_index_map(
            from_index_map(result_obj.objective_order)
            + ["power_production", "power_consumption"]
            + ["alien_power_matrix_cost"]
            + from_index_map(result_obj.extra_cost_order)
            + ["somersloop", "somersloop_usage"]
            + ["item", "resource"]
        )

        variable_breakdowns = {
            variable: create_empty_variable_breakdown(
                variable,
                self.data_parser,
                variable_type_order,
                result_obj.resource_subtype_order
            ) for variable in result_obj.lp_variables
        }

        lp_objective: float = -result_obj.lp_result.fun

        for column_id, column, column_coeff in column_results:
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

        lp_Ax = result_obj.lp_A @ result_obj.lp_result.x

        for variable, breakdown in variable_breakdowns.items():
            if breakdown.type_ not in ["item", "resource"]:
                breakdown.consumption = []

            variable_index = result_obj.lp_variable_indices[variable]
            if variable in result_obj.lp_lower_bounds:
                slack: float = lp_Ax[variable_index] - result_obj.lp_b_l[variable_index]
                if slack < -REPORT_EPSILON:
                    print(f"WARNING: lower bound violation: {variable=} {slack=}")
                breakdown.initial = -result_obj.lp_lower_bounds[variable]
                breakdown.final = slack if abs(slack) > REPORT_EPSILON else 0
            else:
                residual: float = lp_Ax[variable_index] - result_obj.lp_b_l[variable_index]
                if abs(residual) > REPORT_EPSILON:
                    print(f"WARNING: equality constraint violation: {variable=} {residual=}")
            
            self.finalize_variable_budget_side(breakdown.production)
            self.finalize_variable_budget_side(breakdown.consumption)
        
        return column_results, variable_breakdowns, lp_objective

    def finalize_variable_budget_side(self, budget_side: list[BudgetEntry]):
        if not budget_side:
            return
        total_rate = 0.0
        total_count = 0.0
        for budget_entry in budget_side:
            total_rate += budget_entry.rate
            total_count += budget_entry.count
        for budget_entry in budget_side:
            budget_entry.share = budget_entry.rate / total_rate if total_rate else 0.0
        budget_side.sort(key=lambda entry: (-entry.share, entry.desc))
        total = BudgetEntry(desc="Total", count=total_count, rate=total_rate, share=1.0)
        budget_side.insert(0, total)
