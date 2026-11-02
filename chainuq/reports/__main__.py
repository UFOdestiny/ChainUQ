"""Render every generated table, macro file and figure of the paper.

    python -m chainuq.reports
"""

from chainuq.reports import information, results, selective

if __name__ == "__main__":
    results.main()
    selective.main()
    information.main()
