from src.args import get_parser
from src.lp_runner import LPRunner
from pprint import pprint
from src.xlsx_dump import XlxsDump
from typing import Any, cast
import sys

if __name__ == "__main__":
    arg_parser = get_parser()
    args = arg_parser.parse_args()

    runner = LPRunner(args)
    result = runner.run()

    if result["status"] == "error":
        print(f"ERROR: {result['message']}")
        if 'data' in result and result['data']:
            pprint(result['data'])
        sys.exit(1)

    if (args.output_fmt == ""):
        results_data = cast(dict[str, Any], result['data'])

        # Print summary to console
        print("LP result:")
        pprint(results_data['lp_result'])

        print("\nSummary:")
    # print(f"{result['data']['lp_objective']:>17.3f} objective")

        for _, column, column_coeff in results_data['column_results']:
            print(f"{column_coeff:>17.3f} {column.full_display_name}")

        print("\n")

        if args.xlsx_report:
                writer = XlxsDump()
                writer.define(
                    column_results=results_data['column_results'],
                    lp_objective=results_data['lp_objective'],
                    sorted_variable_breakdowns=sorted(results_data['variable_breakdowns'].values(), key=lambda bd: bd.sort_key)
                )
                writer.dump(args)
    # For detailed breakdown, you can inspect result['data']['variable_breakdowns']
    # pprint(result['data']['variable_breakdowns'])


    elif args.output_fmt == 'csv':
         print("Csv data:\n")
         print(result['data'])

