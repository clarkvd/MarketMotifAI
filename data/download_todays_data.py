import datetime
import requests
import pandas as pd
from io import BytesIO

# --- Download News Sentiment Data ---
def download_news_sentiment():
    """
    Downloads the latest News Sentiment data Excel file from the Federal Reserve Bank of San Francisco,
    processes the 'Data' sheet, formats the date and sentiment values, and saves the result as a CSV.

    Output:
        Saves 'news_sentiment_data.csv' in the current directory with formatted columns.
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    url = f"https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx?20240826&{today}"
    print(url)
    output_csv = "news_sentiment_data.csv"

    response = requests.get(url)
    if response.status_code == 200:
        excel_file = BytesIO(response.content)
        df = pd.read_excel(excel_file, sheet_name="Data")
        df['date'] = pd.to_datetime(df['date']).dt.strftime("%-m/%-d/%y")
        df['News Sentiment'] = df['News Sentiment'].round(2)
        df.to_csv(output_csv, index=False)
        print(f"Saved news_sentiment_data.csv for {today}")
    else:
        print(f"Failed to download news sentiment file. Status code: {response.status_code}")

# --- Download CPI Release Dates ---
def download_cpi():
    """
    Downloads the historical release dates for the Consumer Price Index (CPI)
    from the Federal Reserve Bank of St. Louis (ALFRED) in Excel format,
    and saves it as a CSV file.

    Output:
        Saves 'CPI.csv' in the current directory.
    """
    url = "https://alfred.stlouisfed.org/release/downloaddates?rid=10&ff=xlsx"
    output_csv = "CPI.csv"
    response = requests.get(url)
    if response.status_code == 200:
        excel_file = BytesIO(response.content)
        df = pd.read_excel(excel_file, sheet_name="Release Dates")
        df.to_csv(output_csv, index=False)
        print("Saved CPI.csv")
    else:
        print(f"Failed to download CPI. Status code: {response.status_code}")

# --- Download FOMC Release Dates ---
def download_fomc():
    """
    Downloads the historical release dates for Federal Open Market Committee (FOMC)
    decisions from the Federal Reserve Bank of St. Louis (ALFRED) in Excel format,
    and saves it as a CSV file.

    Output:
        Saves 'FOMC.csv' in the current directory.
    """
    url = "https://alfred.stlouisfed.org/release/downloaddates?rid=101&ff=xlsx"
    output_csv = "FOMC.csv"
    response = requests.get(url)
    if response.status_code == 200:
        excel_file = BytesIO(response.content)
        df = pd.read_excel(excel_file, sheet_name="Release Dates")
        df.to_csv(output_csv, index=False)
        print("Saved FOMC.csv")
    else:
        print(f"Failed to download FOMC. Status code: {response.status_code}")

# --- Run All ---
if __name__ == "__main__":
    """
    Executes all data download functions:
        1. News Sentiment data from FRBSF
        2. CPI release dates from ALFRED
        3. FOMC release dates from ALFRED
    """
    download_news_sentiment()
    download_cpi()
    download_fomc()
