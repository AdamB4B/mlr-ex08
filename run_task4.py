# Task 4 sweep: every variant, three seeds, written to results.csv row by row.
#
#   python run_task4.py
#
# Safe to re-run: a (variant, seed) pair that is already in results.csv is skipped, so an
# interrupted sweep continues where it stopped.
import task4_variants as t

if __name__ == "__main__":
    t.run_rows(t.RUN, seeds=t.SEEDS)
