"""An auditable product-analytics and experimentation stack.

Two halves that justify each other. The warehouse half models a real public event stream with dbt
over DuckDB and computes the metrics a product team actually asks for. The experiment half estimates
treatment effects — and then checks its own estimator against data whose true effect is known, which
is the part most A/B write-ups leave out.
"""
__version__ = "0.1.0"
