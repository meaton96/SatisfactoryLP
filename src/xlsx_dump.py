import numpy as np
import json
from datetime import datetime
from .models import LPColumn
from typing import Any
import xlsxwriter


class XlxsDump:

    column_results: list[tuple[str, LPColumn, float]]
    lp_objective: float
    sorted_variable_breakdowns: list[Any]

    def _norm_for_xlsx(self, v):
        # Accept numbers/strings cleanly
        if isinstance(v, (int, float, str)) or v is None:
            return v
        # Numpy scalars
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        # Lists/tuples/sets: join nicely
        if isinstance(v, (list, tuple, set)):
            return ", ".join(self._stringify(x) for x in v)
        # Dicts: JSON so humans can read it in the cell
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        # Everything else: string fallback
        return str(v)

    def _stringify(self, x):
        if isinstance(x, (np.integer, np.floating)):
            return str(float(x))
        return str(x)
    
    def define(self, 
               column_results: list[tuple[str, LPColumn, float]],
               lp_objective: float,
                sorted_variable_breakdowns: list[Any]
               ):
        self.column_results = column_results
        self.lp_objective = lp_objective
        self.sorted_variable_breakdowns = sorted_variable_breakdowns
    
    def dump(self, args):
        print("Writing xlsx report")

        

        workbook = xlsxwriter.Workbook(f'reports/{args.xlsx_report}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.xlsx', 
                                    {"nan_inf_to_errors": True})

        default_format = workbook.add_format({"align": "center"})
        top_format = workbook.add_format({"align": "center", "top": True})
        bold_format = workbook.add_format({"align": "center", "bold": True})
        bold_underline_format = workbook.add_format(
            {"align": "center", "bold": True, "underline": True}
        )
        bold_top_format = workbook.add_format(
            {"align": "center", "bold": True, "top": True}
        )
        bold_underline_top_format = workbook.add_format(
            {"align": "center", "bold": True, "underline": True, "top": True}
        )
        percent_format = workbook.add_format({"align": "center", "num_format": "0.0#####%"})

        sheet_breakdown = workbook.add_worksheet("Breakdown" + args.xlsx_sheet_suffix)
        sheet_list = workbook.add_worksheet("List" + args.xlsx_sheet_suffix)
        sheet_config = workbook.add_worksheet("Config" + args.xlsx_sheet_suffix)

        def write_cell(sheet, row, col, value, fmt=None):
            value = self._norm_for_xlsx(value)
            if fmt is None:
                sheet.write(row, col, value)
            else:
                sheet.write(row, col, value, fmt)


        sheet_list.add_table(
            0,
            0,
            len(self.column_results),
            6,
            {
                "columns": [
                    {"header": header, "header_format": bold_format}
                    for header in [
                        "Type",
                        "Name",
                        "Machine",
                        "Subtype",
                        "Clock",
                        "Somersloops",
                        "Quantity",
                    ]
                ],
                "style": "Table Style Light 16",
            },
        )

        write_cell(sheet_list, 1, 0, "objective")
        write_cell(sheet_list, 1, 1, "objective")
        write_cell(sheet_list, 1, 6, self.lp_objective)

        for i, (column_id, column, column_coeff) in enumerate(self.column_results):
            write_cell(sheet_list, 2 + i, 0, column.type_)
            write_cell(sheet_list, 2 + i, 1, column.display_name)
            write_cell(sheet_list, 2 + i, 2, column.machine_name)
            write_cell(sheet_list, 2 + i, 3, column.resource_subtype)
            write_cell(sheet_list, 2 + i, 4, column.clock, fmt=percent_format)
            write_cell(sheet_list, 2 + i, 5, column.somersloops)
            write_cell(sheet_list, 2 + i, 6, column_coeff)

        for c, width in enumerate([19, 39, 25, 19, 11, 17, 13]):
            sheet_list.set_column(c, c, width)

        current_row = 0
        max_budget_entries = 0
        budget_rows = [
            ("desc", "Producer", "Consumer"),
            ("count", "Producer Count", "Consumer Count"),
            ("rate", "Production Rate", "Consumption Rate"),
            ("share", "Production Share", "Consumption Share"),
        ]

        production_share_cf = {
            "type": "2_color_scale",
            "min_type": "num",
            "max_type": "num",
            "min_value": 0,
            "max_value": 1,
            "min_color": "#FFFFFF",
            "max_color": "#99FF99",
        }
        consumption_share_cf = production_share_cf.copy()
        consumption_share_cf["max_color"] = "#FFCC66"

        for variable_index, breakdown in enumerate(self.sorted_variable_breakdowns):
            for budget_side_index, budget_side_name, budget_side in (
                (0, "production", breakdown.production),
                (1, "consumption", breakdown.consumption),
            ):
                if not budget_side:
                    continue
                for budget_row in budget_rows:
                    key = budget_row[0]
                    name = budget_row[budget_side_index + 1]
                    if key == "desc":
                        fmts = (bold_top_format, bold_underline_top_format)
                    elif key == "share":
                        fmts = (bold_format, percent_format)
                    else:
                        fmts = (bold_format, default_format)
                    write_cell(
                        sheet_breakdown, current_row, 0, breakdown.display_name, fmt=fmts[0]
                    )
                    write_cell(sheet_breakdown, current_row, 1, name, fmt=fmts[0])
                    for i, entry in enumerate(budget_side):
                        value = getattr(entry, key)
                        write_cell(sheet_breakdown, current_row, 2 + i, value, fmt=fmts[1])
                    if key == "share":
                        cf = (
                            production_share_cf
                            if budget_side_name == "production"
                            else consumption_share_cf
                        )
                        sheet_breakdown.conditional_format(
                            current_row, 3, current_row, len(budget_side) + 1, cf
                        )
                    max_budget_entries = max(max_budget_entries, len(budget_side))
                    current_row += 1

            for initial_or_final, initial_or_final_value in (
                ("initial", breakdown.initial),
                ("final", breakdown.final),
            ):
                if initial_or_final_value is None:
                    continue
                if initial_or_final == "initial":
                    fmts = (bold_top_format, top_format)
                else:
                    fmts = (bold_format, default_format)
                write_cell(
                    sheet_breakdown, current_row, 0, breakdown.display_name, fmt=fmts[0]
                )
                write_cell(
                    sheet_breakdown,
                    current_row,
                    1,
                    initial_or_final.capitalize(),
                    fmt=fmts[0],
                )
                write_cell(
                    sheet_breakdown, current_row, 2, initial_or_final_value, fmt=fmts[1]
                )
                current_row += 1

        for c, width in enumerate([41, 19, 13] + [59] * (max_budget_entries - 1)):
            sheet_breakdown.set_column(c, c, width)

        for i, (arg_name, arg_value) in enumerate(vars(args).items()):
            write_cell(sheet_config, i, 0, arg_name)
            write_cell(sheet_config, i, 1, arg_value)

        for c, width in enumerate([36, 19]):
            sheet_config.set_column(c, c, width)

        workbook.close()