# FPTU_DPL302m_LAB4
Prediction Stocks use Machine Learning

# Download data
git clone ....

# Run dockerfile

# Using terminal for run file
    parser.add_argument("--root", '-r', type=str, default="data", help="Path to your data storage")
    parser.add_argument("--ticker", '-t', type=str, default="NVDA", help="Default: NVDA")
    parser.add_argument("--start_date", '-s', type=str, default="2020-01-01", help="Start_date download")
    parser.add_argument("--end_date", '-e', type=str, default=datetime.now().strftime('%Y-%m-%d'), help="End_date download")

COPY THIS ONE AND PASTE ON TERMINAL TO RUN :  python local_stock.py --ticker NVDA --start_date 1980-10-28 

# Explain: --ticker : you can easy change ticker stock you want, eg: "NVDA" , "TSLA", "GOOG", "META", "AAPL"

