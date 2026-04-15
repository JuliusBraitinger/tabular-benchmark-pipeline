# This file makes 'pipeline/data/' a Python package.
# It also re-exports Dataset and DataSource so you can write:
#   from pipeline.data import Dataset, DataSource
# instead of:
#   from pipeline.data.base import Dataset, DataSource

from pipeline.data.base import Dataset, DataSource

__all__ = ["Dataset", "DataSource"]
