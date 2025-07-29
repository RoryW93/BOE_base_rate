"""
This script processes the Bank of England base rate data - starting from 2004 to present

# The bank rate time-series is visualised
# The histogram of the time-series bank rate data is assessed
# The PDF is parametrically estimated

See processing_functions module for details of processing routines

Date: 14/06/25
Author: Rory White
Location: Nottingham, UK
"""

from processing_functions import *

# (1) pre process data
data = pre_process_data()

# (2) extract statistical summary
print("Full dataset:\n" + str(data["Rate"].describe(())))
print(f"Latest Bank Of England base rate: " + str(data["Date Changed"].iloc[-1]) + ", " + str(data["Rate"].iloc[-1]) + "%")

# (3) Visualise full time-series data
vis_full_dataset(data)

# (4) reference the last time the base rate was equivalent in value - past timedate
data_comparison = compare_data(data)

# (5) Filter last 20 years of data
data_from04 = filter_data_2004(data)

# (6) estimate probability distribution function
vals, probs = estimate_PDF(data)
vals04, probs04 = estimate_PDF(data_from04)

# (7) Visualise statistical distributions
vis_stat_profile(data)
vis_stat_profile(data_from04)


