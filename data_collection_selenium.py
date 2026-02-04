"""
Data collection script using Selenium: Collect Canadian Government bond data from Business Insider
Date range: January 5-19, 2026 (10 weekdays)
"""

import sys
import os

def check_environment():
    """Check if required packages are available"""
    try:
        import selenium
        return
    except ImportError:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        venv_python = os.path.join(script_dir, 'venv', 'bin', 'python3')
        
        if os.path.exists(venv_python):
            print("=" * 60)
            print("Not running in virtual environment")
            print("Attempting to use virtual environment Python...")
            print("=" * 60)
            import subprocess
            result = subprocess.run([venv_python, __file__] + sys.argv[1:])
            sys.exit(result.returncode)
        else:
            print("=" * 60)
            print("ERROR: selenium module not found")
            print("=" * 60)
            print("\nPlease follow these steps:")
            print("1. Activate virtual environment:")
            print("   source venv/bin/activate")
            print("\n2. If virtual environment doesn't exist, create and install dependencies:")
            print("   python3 -m venv venv")
            print("   source venv/bin/activate")
            print("   pip install -r requirements.txt")
            print("\n3. Then run the script:")
            print("   python data_collection_selenium.py")
            print("=" * 60)
            sys.exit(1)

check_environment()

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time
import json
from datetime import datetime, timedelta
import re
import requests
from urllib.parse import unquote, quote

SHORT_TERM_URL = "https://markets.businessinsider.com/bonds/finder?borrower=71&maturity=shortterm&yield=&bondtype=2%2c3%2c4%2c16&coupon=&currency=184&rating=&country=19"
MID_TERM_URL = "https://markets.businessinsider.com/bonds/finder?borrower=71&maturity=midterm&yield=&bondtype=2%2c3%2c4%2c16&coupon=&currency=184&rating=&country=19"

START_DATE = datetime(2026, 1, 5)
END_DATE = datetime(2026, 1, 19)

def get_weekdays(start_date, end_date):
    """Generate all weekdays between start and end dates"""
    weekdays = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday to Friday
            weekdays.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    return weekdays

def setup_driver(headless=True):
    """Setup Selenium WebDriver"""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            '''
        })
        return driver
    except Exception as e:
        print(f"Failed to start Chrome WebDriver: {e}")
        print("Please ensure Chrome and ChromeDriver are installed")
        print("macOS installation: brew install chromedriver")
        return None

def fetch_bond_list(driver, url):
    """Fetch bond list from Business Insider"""
    print(f"Fetching bond list: {url}")
    bonds = []
    
    try:
        driver.get(url)
        time.sleep(3)
        
        wait = WebDriverWait(driver, 20)
        selectors = [
            "table",
            ".table",
            "[class*='bond']",
            "[class*='table']",
            "tbody tr"
        ]
        
        table = None
        for selector in selectors:
            try:
                table = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                break
            except TimeoutException:
                continue
        
        if table:
            rows = driver.find_elements(By.CSS_SELECTOR, "tbody tr, tr[class*='row'], tr[class*='bond']")
            
            for row in rows:
                try:
                    link_elem = row.find_element(By.CSS_SELECTOR, "a[href*='/bonds/']")
                    if link_elem:
                        bond_url = link_elem.get_attribute('href')
                        if not bond_url.startswith('http'):
                            bond_url = 'https://markets.businessinsider.com' + bond_url
                        
                        bond_name = link_elem.text.strip()
                        
                        maturity = None
                        try:
                            maturity_elem = row.find_element(By.CSS_SELECTOR, "td:nth-child(3), [class*='maturity']")
                            maturity = maturity_elem.text.strip()
                        except:
                            pass
                        
                        bonds.append({
                            'name': bond_name,
                            'url': bond_url,
                            'maturity': maturity
                        })
                except NoSuchElementException:
                    continue
        
        print(f"Found {len(bonds)} bonds")
        
    except Exception as e:
        print(f"Error fetching bond list: {e}")
        import traceback
        traceback.print_exc()
    
    return bonds

def fetch_bond_details(driver, bond_url):
    """Fetch detailed information for a single bond (Snapshot page)"""
    print(f"Fetching bond details: {bond_url}")
    
    bond_info = {
        'url': bond_url,
        'name': None,
        'coupon': None,
        'isin': None,
        'issue_date': None,
        'maturity_date': None,
        'exchange': None,
        'issue_price': None,
        'currency': None
    }
    
    try:
        driver.get(bond_url)
        time.sleep(4)
        
        wait = WebDriverWait(driver, 20)
        
        table = None
        table_selectors = [
            "table.table.table--no-vertical-border",
            "table.table",
            "table[data-table-filter-table]",
            "table",
            ".table"
        ]
        
        for selector in table_selectors:
            try:
                table = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                if table:
                    print(f"  Found table: {selector}")
                    break
            except TimeoutException:
                continue
        
        if not table:
            try:
                h2_elem = driver.find_element(By.XPATH, "//h2[contains(text(), 'Bond Data')]")
                table = h2_elem.find_element(By.XPATH, "./following::table[1]")
                print("  Found table via Bond Data heading")
            except:
                print("  Warning: Data table not found, attempting text extraction")
                page_text = driver.find_element(By.TAG_NAME, "body").text
                return extract_from_text(page_text, bond_info, bond_url)
        
        if not table:
            print("  Error: Cannot find data table")
            return bond_info
        
        driver.execute_script("arguments[0].scrollIntoView(true);", table)
        time.sleep(1)
        rows = []
        row_selectors = [
            "tbody tr.table_tr",
            "tbody tr",
            "tr.table_tr",
            "tr"
        ]
        
        for selector in row_selectors:
            try:
                rows = table.find_elements(By.CSS_SELECTOR, selector)
                if rows:
                    print(f"  Found {len(rows)} rows (using selector: {selector})")
                    break
            except:
                continue
        
        if not rows:
            print("  Warning: No table rows found")
            return bond_info
        
        data_dict = {}
        
        for i, row in enumerate(rows):
            try:
                cells = []
                cell_selectors = [
                    "td.table_td",
                    "td",
                    "th, td"
                ]
                
                for cell_selector in cell_selectors:
                    try:
                        cells = row.find_elements(By.CSS_SELECTOR, cell_selector)
                        if len(cells) >= 2:
                            break
                    except:
                        continue
                
                if len(cells) >= 2:
                    label = cells[0].text.strip()
                    value = cells[1].text.strip()
                    if label and value:
                        data_dict[label] = value
            except Exception as e:
                continue
        
        if data_dict:
            print(f"  Extracted {len(data_dict)} fields from table: {list(data_dict.keys())[:5]}...")
        else:
            print("  Warning: Failed to extract any data from table")
            try:
                page_source = driver.page_source
                with open(f'debug_page_{bond_url.split("/")[-1]}.html', 'w', encoding='utf-8') as f:
                    f.write(page_source)
                print(f"  Saved page HTML to debug_page_{bond_url.split('/')[-1]}.html")
            except:
                pass
        
        data_dict_lower = {k.lower(): v for k, v in data_dict.items()}
        
        # ISIN
        for key in ['isin', 'isincode']:
            if key in data_dict_lower:
                bond_info['isin'] = data_dict_lower[key].strip()
                break
        
        # Name
        for key in ['name', 'bond name', 'security name']:
            if key in data_dict_lower:
                bond_info['name'] = data_dict_lower[key].strip()
                break
        
        for key in ['coupon', 'coupon rate', 'interest rate']:
            if key in data_dict_lower:
                coupon_text = data_dict_lower[key].strip()
                coupon_match = re.search(r'([\d.]+)', coupon_text)
                if coupon_match:
                    bond_info['coupon'] = coupon_match.group(1)
                    break
        
        # Issue Date
        for key in ['issue date', 'issued', 'issue']:
            if key in data_dict_lower:
                bond_info['issue_date'] = data_dict_lower[key].strip()
                break
        
        # Maturity Date
        for key in ['maturity date', 'maturity', 'matures']:
            if key in data_dict_lower:
                bond_info['maturity_date'] = data_dict_lower[key].strip()
                break
        
        for key in ['exchange', 'traded on', 'market', 'trading venue']:
            if key in data_dict_lower:
                bond_info['exchange'] = data_dict_lower[key].strip()
                break
        
        # Issue Price
        for key in ['issue price', 'price', 'issue']:
            if key in data_dict_lower:
                issue_price_text = data_dict_lower[key].strip()
                price_match = re.search(r'([\d.]+)', issue_price_text)
                if price_match:
                    bond_info['issue_price'] = price_match.group(1)
                    break
        
        # Currency
        for key in ['currency', 'curr']:
            if key in data_dict_lower:
                bond_info['currency'] = data_dict_lower[key].strip()
                break
        
    except Exception as e:
        print(f"Error fetching bond details: {e}")
        import traceback
        traceback.print_exc()
    
    return bond_info

def extract_from_text(page_text, bond_info, bond_url):
    """Extract information from page text (fallback method)"""
    # ISIN
    isin_match = re.search(r'ISIN[:\s]+([A-Z0-9]{12})', page_text, re.IGNORECASE)
    if isin_match:
        bond_info['isin'] = isin_match.group(1)
    
    # Coupon
    coupon_match = re.search(r'Coupon[:\s]+([\d.]+)', page_text, re.IGNORECASE)
    if coupon_match:
        bond_info['coupon'] = coupon_match.group(1)
    
    # Issue Date
    issue_match = re.search(r'Issue\s+Date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{4})', page_text, re.IGNORECASE)
    if issue_match:
        bond_info['issue_date'] = issue_match.group(1)
    
    # Maturity Date
    maturity_match = re.search(r'Maturity\s+Date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{4})', page_text, re.IGNORECASE)
    if maturity_match:
        bond_info['maturity_date'] = maturity_match.group(1)
    
    return bond_info

def extract_tkdata_from_page(driver, bond_url):
    """
    Extract tkData identifier from bond page
    tkData is used to call Chart_GetChartData API for historical prices
    """
    try:
        time.sleep(2)
        
        try:
            WebDriverWait(driver, 5).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
        except:
            pass
        
        page_source = driver.page_source
        
        chart_url_pattern = r'Chart_GetChartData[^"\']*tkData=([^"\'&]+)'
        match = re.search(chart_url_pattern, page_source, re.IGNORECASE)
        if match:
            tkdata_encoded = match.group(1)
            tkdata = unquote(tkdata_encoded)
            print(f"  Found tkData from Chart URL: {tkdata[:50]}...")
            return tkdata
        
        js_patterns = [
            r'tkData["\']?\s*[:=]\s*["\']([^"\']+)["\']',
            r'tkData["\']?\s*=\s*["\']([^"\']+)["\']',
            r'["\']tkData["\']\s*:\s*["\']([^"\']+)["\']',
            r'tkData\s*[:=]\s*([0-9,]+)',
        ]
        
        for pattern in js_patterns:
            matches = re.finditer(pattern, page_source, re.IGNORECASE)
            for match in matches:
                tkdata = match.group(1)
                if '%' in tkdata:
                    tkdata = unquote(tkdata)
                if re.match(r'^[\d,]+$', tkdata.replace(' ', '')):
                    print(f"  Found tkData from JavaScript variable: {tkdata[:50]}...")
                    return tkdata
        try:
            scripts = driver.find_elements(By.TAG_NAME, "script")
            for script in scripts:
                script_text = script.get_attribute('innerHTML') or script.get_attribute('textContent') or ''
                if not script_text:
                    continue
                
                patterns = [
                    r'tkData["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                    r'tkData\s*[:=]\s*([0-9,]+)',
                    r'Chart_GetChartData[^"\']*tkData=([^"\'&]+)',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, script_text, re.IGNORECASE)
                    if match:
                        tkdata = match.group(1)
                        if '%' in tkdata:
                            tkdata = unquote(tkdata)
                        if re.match(r'^[\d,]+$', tkdata.replace(' ', '')):
                            print(f"  Found tkData from script tag: {tkdata[:50]}...")
                            return tkdata
        except Exception:
            pass
        
        print("  Warning: Cannot extract tkData from page")
        print("  Hint: May need to check if page is fully loaded or if page structure is different")
        return None
        
    except Exception as e:
        print(f"  Error extracting tkData: {e}")
        return None

def fetch_historical_prices_via_api(bond_url, tkdata, dates, max_retries=3):
    """
    Fetch historical price data using Chart_GetChartData API
    This is a more reliable method
    """
    import requests
    from urllib.parse import quote
    
    prices = {}
    
    if not tkdata:
        print("  Cannot fetch historical prices: Missing tkData")
        return prices
    
    try:
        from_date = dates[0].replace('-', '') if dates else '20260105'
        to_date = dates[-1].replace('-', '') if dates else '20260119'
        
        tkdata_encoded = quote(tkdata, safe='')
        
        api_url = (
            f"https://markets.businessinsider.com/Ajax/Chart_GetChartData"
            f"?instrumentType=Bond&tkData={tkdata_encoded}&from={from_date}&to={to_date}"
        )
        
        print(f"  Calling API: {api_url[:100]}...")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": bond_url,
            "X-Requested-With": "XMLHttpRequest"
        }
        
        response = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = (attempt + 1) * 3
                    print(f"  Waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                
                response = requests.get(api_url, headers=headers, timeout=60)
                response.raise_for_status()
                break
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    continue
                else:
                    print(f"  API request timeout, retried {max_retries} times, giving up")
                    return prices
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 503:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        print(f"  Server 503 error, waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  Server 503 error, retried {max_retries} times, giving up")
                        return prices
                else:
                    print(f"  HTTP error: {e}")
                    return prices
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    continue
                else:
                    print(f"  API request failed: {e}")
                    return prices
        
        if response is None:
            return prices
        
        data = response.json()
        
        def extract_time_series_points(obj, points=None):
            """Recursively extract time series points [timestamp, value]"""
            if points is None:
                points = []
            
            if isinstance(obj, list):
                if len(obj) >= 2 and all(isinstance(x, (int, float)) for x in obj[:2]):
                    points.append((obj[0], obj[1]))
                else:
                    for item in obj:
                        extract_time_series_points(item, points)
            elif isinstance(obj, dict):
                keys_lower = {str(k).lower(): k for k in obj.keys()}
                
                time_key = None
                for tkey in ['x', 't', 'time', 'date', 'timestamp']:
                    if tkey in keys_lower:
                        time_key = keys_lower[tkey]
                        break
                
                value_key = None
                for vkey in ['y', 'v', 'value', 'close', 'price', 'last']:
                    if vkey in keys_lower:
                        value_key = keys_lower[vkey]
                        break
                
                if time_key and value_key:
                    points.append((obj[time_key], obj[value_key]))
                else:
                    for value in obj.values():
                        extract_time_series_points(value, points)
            
            return points
        
        time_series_points = extract_time_series_points(data)
        
        if not time_series_points:
            print("  Warning: No time series points found in API response")
            return prices
        
        print(f"  Extracted {len(time_series_points)} data points from API")
        
        for timestamp, value in time_series_points:
            try:
                date_obj = None
                
                if isinstance(timestamp, str):
                    try:
                        date_obj = datetime.strptime(timestamp, '%Y-%m-%d')
                    except:
                        try:
                            date_obj = pd.to_datetime(timestamp)
                        except:
                            continue
                elif isinstance(timestamp, (int, float)):
                    if timestamp > 1e11:
                        date_obj = datetime.fromtimestamp(timestamp / 1000)
                    else:
                        date_obj = datetime.fromtimestamp(timestamp)
                else:
                    continue
                
                if date_obj is None:
                    continue
                
                date_str = date_obj.strftime('%Y-%m-%d')
                
                price = float(value)
                
                if not (50 < price < 150):
                    continue
                target_start = datetime.strptime(dates[0], '%Y-%m-%d').date()
                target_end = datetime.strptime(dates[-1], '%Y-%m-%d').date()
                date_only = date_obj.date()
                
                if date_str in dates or (target_start <= date_only <= target_end):
                    prices[date_str] = price
                    
            except (ValueError, TypeError, OSError) as e:
                continue
        
        print(f"  Successfully extracted {len(prices)} price data points")
        
    except requests.RequestException as e:
        print(f"  API request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"  JSON parsing failed: {e}")
    except Exception as e:
        print(f"  Error fetching historical prices: {e}")
        import traceback
        traceback.print_exc()
    
    return prices

def fetch_historical_prices(driver, bond_url, dates):
    """
    Fetch historical price data for a bond
    Uses API method (more reliable)
    """
    print(f"Fetching historical prices: {bond_url}")
    
    prices = {}
    
    try:
        if driver.current_url != bond_url:
            driver.get(bond_url)
            time.sleep(4)
        
        tkdata = None
        max_attempts = 2
        
        for attempt in range(max_attempts):
            tkdata = extract_tkdata_from_page(driver, bond_url)
            if tkdata:
                break
            elif attempt < max_attempts - 1:
                print(f"  Attempt {attempt + 1} failed, waiting before retry...")
                time.sleep(2)
                driver.refresh()
                time.sleep(3)
        
        if tkdata:
            prices = fetch_historical_prices_via_api(bond_url, tkdata, dates)
        else:
            print("  Warning: Cannot extract tkData, skipping price data for this bond")
        
    except Exception as e:
        print(f"Error fetching historical prices: {e}")
        import traceback
        traceback.print_exc()
    
    return prices

def normalize_date(date_str):
    """Normalize date format to YYYY-MM-DD"""
    formats = [
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y/%m/%d',
        '%B %d, %Y',
        '%b %d, %Y',
        '%d %B %Y',
        '%d %b %Y'
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime('%Y-%m-%d')
        except:
            continue
    
    return date_str

def filter_bonds_by_maturity(bonds_data, max_years=10):
    """Filter bonds: only keep bonds with maturity < 10 years (from Jan 5, 2026)"""
    filtered = []
    reference_date = datetime(2026, 1, 5)
    
    for bond in bonds_data:
        maturity_date = bond.get('maturity_date')
        if not maturity_date:
            continue
        
        try:
            maturity_dt = None
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    maturity_dt = datetime.strptime(maturity_date, fmt)
                    break
                except:
                    continue
            
            if maturity_dt:
                years_to_maturity = (maturity_dt - reference_date).days / 365.25
                if 0 < years_to_maturity < max_years:
                    filtered.append(bond)
        except:
            continue
    
    return filtered

def collect_all_data(headless=True):
    """Main function to collect all data"""
    print("=" * 60)
    print("Starting Canadian Government Bond Data Collection")
    print("=" * 60)
    
    today = datetime.now()
    if START_DATE > today:
        print(f"Warning: Start date {START_DATE.strftime('%Y-%m-%d')} is in the future")
        print("If data doesn't exist, may need to use historical data or wait until specified date")
        print()
    
    weekdays = get_weekdays(START_DATE, END_DATE)
    print(f"Need to collect price data for {len(weekdays)} weekdays")
    print(f"Date range: {weekdays[0]} to {weekdays[-1]}")
    print()
    
    driver = setup_driver(headless=headless)
    if not driver:
        print("Failed to start WebDriver, please check Chrome and ChromeDriver installation")
        return []
    
    try:
        all_bonds = []
        
        for url in [SHORT_TERM_URL, MID_TERM_URL]:
            bonds = fetch_bond_list(driver, url)
            all_bonds.extend(bonds)
            time.sleep(2)
        
        seen_urls = set()
        unique_bonds = []
        for bond in all_bonds:
            if bond['url'] not in seen_urls:
                seen_urls.add(bond['url'])
                unique_bonds.append(bond)
        
        print(f"\nFound {len(unique_bonds)} unique bonds")
        
        bonds_data = []
        for i, bond in enumerate(unique_bonds, 1):
            print(f"\nProcessing bond {i}/{len(unique_bonds)}: {bond.get('name', 'Unknown')}")
            
            details = fetch_bond_details(driver, bond['url'])
            if details:
                prices = fetch_historical_prices(driver, bond['url'], weekdays)
                details['historical_prices'] = prices
                details['price_count'] = len(prices)
                bonds_data.append(details)
                
                print(f"  Extracted data: coupon={details.get('coupon')}, isin={details.get('isin')}, "
                      f"issue_date={details.get('issue_date')}, maturity_date={details.get('maturity_date')}, "
                      f"exchange={details.get('exchange')}, price_count={len(prices)}")
            
            time.sleep(3)
        
        print(f"\nCollected raw data for {len(bonds_data)} bonds (unfiltered)")
        
        output_file = 'bonds_data_raw.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(bonds_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nRaw data saved to {output_file}")
        
        filtered_bonds = filter_bonds_by_maturity(bonds_data, max_years=10)
        print(f"After filtering: {len(filtered_bonds)} bonds (maturity < 10 years)")
        
        output_file_filtered = 'bonds_data.json'
        with open(output_file_filtered, 'w', encoding='utf-8') as f:
            json.dump(filtered_bonds, f, indent=2, ensure_ascii=False)
        
        print(f"Filtered data saved to {output_file_filtered}")
        
        bonds_data_for_csv = bonds_data
        df_data = []
        for bond in bonds_data_for_csv:
            if bond.get('historical_prices'):
                for date, price in bond.get('historical_prices', {}).items():
                    df_data.append({
                        'ISIN': bond.get('isin', ''),
                        'Coupon': bond.get('coupon', ''),
                        'Issue_Date': bond.get('issue_date', ''),
                        'Maturity_Date': bond.get('maturity_date', ''),
                        'Name': bond.get('name', ''),
                        'URL': bond.get('url', ''),
                        'Exchange': bond.get('exchange', ''),
                        'Date': date,
                        'Close_Price': price
                    })
            else:
                df_data.append({
                    'ISIN': bond.get('isin', ''),
                    'Coupon': bond.get('coupon', ''),
                    'Issue_Date': bond.get('issue_date', ''),
                    'Maturity_Date': bond.get('maturity_date', ''),
                    'Name': bond.get('name', ''),
                    'URL': bond.get('url', ''),
                    'Exchange': bond.get('exchange', ''),
                    'Date': '',
                    'Close_Price': ''
                })
        
        if df_data:
            df = pd.DataFrame(df_data)
            csv_file = 'bonds_data_raw.csv'
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"Raw data CSV saved to {csv_file}")
        
        return bonds_data_for_csv
        
    finally:
        driver.quit()
        print("\nWebDriver closed")

if __name__ == "__main__":
    headless = True
    if len(sys.argv) > 1 and sys.argv[1] == '--no-headless':
        headless = False
        print("Using non-headless mode (for debugging)")
    
    data = collect_all_data(headless=headless)
    print(f"\nCompleted! Collected data for {len(data)} bonds")
