"""
Quantize a model to EXL3 format.
Usage: python quantize.py --in_dir <model> --out_dir <output> --work_dir <work> --bits <bpw>
"""
import sys
from exllamav3.conversion.convert_model import parser, prepare, main

if __name__ == "__main__":
    parsed = parser.parse_args()
    args, job_state, success, error = prepare(parsed)
    if not success:
        print(f"Error: {error}")
        sys.exit(1)
    main(args, job_state)
