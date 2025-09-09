from src.args import get_parser
from src.lp_runner import LPRunner
from pprint import pprint
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

    # Print summary to console
    print("LP result:")
    pprint(result['data']['lp_result'])

    print("\nSummary:")
    print(f"{result['data']['lp_objective']:>17.3f} objective")

    for _, column, column_coeff in result['data']['column_results']:
        print(f"{column_coeff:>17.3f} {column.full_display_name}")

    print("\n")
    # For detailed breakdown, you can inspect result['data']['variable_breakdowns']
    # pprint(result['data']['variable_breakdowns'])

