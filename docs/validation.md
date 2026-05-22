# Validation

Financial ML systems fail when evaluation is allowed to see the future. This lab starts with a minimal time-ordered split and a transaction-cost-aware baseline before adding model complexity.

Public examples use synthetic data only.

Validation principles:

- preserve time order
- compare against a no-ML baseline
- include transaction costs
- report drawdown, turnover, and Sharpe
- keep production strategy parameters private
