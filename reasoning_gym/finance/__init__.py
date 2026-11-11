"""
Finance tasks for training reasoning capabilities:

Procedural datasets covering financial metrics and calculations. The first
dataset, ``fund_ratios``, generates private-equity / venture-capital fund
performance multiple problems (DPI, RVPI, TVPI, MOIC).
"""

from .fund_ratios import FundRatiosConfig, FundRatiosCurriculum, FundRatiosDataset

__all__ = ["FundRatiosConfig", "FundRatiosDataset", "FundRatiosCurriculum"]
