"""
数据收集脚本（使用Selenium）：从Business Insider收集加拿大政府债券数据
日期范围：2026年1月5日 - 2026年1月19日（10个工作日）
"""

import sys
import os

# 检查是否在虚拟环境中，如果不是，尝试自动使用虚拟环境
def check_environment():
    """检查运行环境，如果不在虚拟环境中，尝试使用虚拟环境的Python"""
    try:
        import selenium
        # 成功导入，继续执行
        return
    except ImportError:
        # 未找到selenium，尝试使用虚拟环境
        script_dir = os.path.dirname(os.path.abspath(__file__))
        venv_python = os.path.join(script_dir, 'venv', 'bin', 'python3')
        
        if os.path.exists(venv_python):
            print("=" * 60)
            print("检测到未在虚拟环境中运行")
            print("正在尝试使用虚拟环境中的Python...")
            print("=" * 60)
            # 使用虚拟环境的Python重新运行脚本
            import subprocess
            result = subprocess.run([venv_python, __file__] + sys.argv[1:])
            sys.exit(result.returncode)
        else:
            print("=" * 60)
            print("错误: 未找到selenium模块")
            print("=" * 60)
            print("\n请按照以下步骤操作:")
            print("1. 激活虚拟环境:")
            print("   source venv/bin/activate")
            print("\n2. 如果虚拟环境不存在，创建并安装依赖:")
            print("   python3 -m venv venv")
            print("   source venv/bin/activate")
            print("   pip install -r requirements.txt")
            print("\n3. 然后运行脚本:")
            print("   python data_collection_selenium.py")
            print("\n或者使用提供的运行脚本:")
            print("   ./run_data_collection.sh")
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

# 数据源URL
SHORT_TERM_URL = "https://markets.businessinsider.com/bonds/finder?borrower=71&maturity=shortterm&yield=&bondtype=2%2c3%2c4%2c16&coupon=&currency=184&rating=&country=19"
MID_TERM_URL = "https://markets.businessinsider.com/bonds/finder?borrower=71&maturity=midterm&yield=&bondtype=2%2c3%2c4%2c16&coupon=&currency=184&rating=&country=19"

# 日期范围
START_DATE = datetime(2026, 1, 5)
END_DATE = datetime(2026, 1, 19)

def get_weekdays(start_date, end_date):
    """生成从开始日期到结束日期之间的所有工作日"""
    weekdays = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday to Friday
            weekdays.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    return weekdays

def setup_driver(headless=True):
    """设置Selenium WebDriver"""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument('--headless')  # 无头模式
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
        # 执行脚本以隐藏webdriver特征
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            '''
        })
        return driver
    except Exception as e:
        print(f"无法启动Chrome WebDriver: {e}")
        print("请确保已安装Chrome和ChromeDriver")
        print("macOS安装方法: brew install chromedriver")
        return None

def fetch_bond_list(driver, url):
    """从Business Insider获取债券列表"""
    print(f"正在获取债券列表: {url}")
    bonds = []
    
    try:
        driver.get(url)
        time.sleep(3)  # 等待页面加载
        
        # 等待表格加载
        wait = WebDriverWait(driver, 20)
        
        # 尝试多种可能的选择器来找到债券表格
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
            # 查找所有债券行
            rows = driver.find_elements(By.CSS_SELECTOR, "tbody tr, tr[class*='row'], tr[class*='bond']")
            
            for row in rows:
                try:
                    # 查找债券链接
                    link_elem = row.find_element(By.CSS_SELECTOR, "a[href*='/bonds/']")
                    if link_elem:
                        bond_url = link_elem.get_attribute('href')
                        if not bond_url.startswith('http'):
                            bond_url = 'https://markets.businessinsider.com' + bond_url
                        
                        bond_name = link_elem.text.strip()
                        
                        # 提取到期日期（如果可见）
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
        
        print(f"找到 {len(bonds)} 个债券")
        
    except Exception as e:
        print(f"获取债券列表时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return bonds

def fetch_bond_details(driver, bond_url):
    """获取单个债券的详细信息（Snapshot页面）"""
    print(f"正在获取债券详情: {bond_url}")
    
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
        time.sleep(4)  # 增加等待时间，确保页面完全加载
        
        wait = WebDriverWait(driver, 20)
        
        # 尝试多种方式查找数据表格
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
                    print(f"  找到表格: {selector}")
                    break
            except TimeoutException:
                continue
        
        if not table:
            # 如果找不到表格，尝试查找包含"Bond Data"标题的区域
            try:
                h2_elem = driver.find_element(By.XPATH, "//h2[contains(text(), 'Bond Data')]")
                # 找到h2后面的表格
                table = h2_elem.find_element(By.XPATH, "./following::table[1]")
                print("  通过Bond Data标题找到表格")
            except:
                print("  警告: 未找到数据表格，尝试从页面文本提取")
                # 如果还是找不到，尝试从整个页面文本提取
                page_text = driver.find_element(By.TAG_NAME, "body").text
                # 使用正则表达式作为备选方案
                return extract_from_text(page_text, bond_info, bond_url)
        
        if not table:
            print("  错误: 无法找到数据表格")
            return bond_info
        
        # 滚动到表格位置，确保可见
        driver.execute_script("arguments[0].scrollIntoView(true);", table)
        time.sleep(1)
        
        # 提取所有表格行 - 尝试多种选择器
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
                    print(f"  找到 {len(rows)} 行 (使用选择器: {selector})")
                    break
            except:
                continue
        
        if not rows:
            print("  警告: 未找到表格行")
            return bond_info
        
        # 创建一个字典来存储标签-值对
        data_dict = {}
        
        for i, row in enumerate(rows):
            try:
                # 每行有两个td：第一个是标签，第二个是值
                # 尝试多种td选择器
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
        
        # 调试：打印提取到的所有字段
        if data_dict:
            print(f"  从表格提取到 {len(data_dict)} 个字段: {list(data_dict.keys())[:5]}...")
        else:
            print("  警告: 未能从表格提取到任何数据")
            # 尝试保存页面HTML用于调试
            try:
                page_source = driver.page_source
                with open(f'debug_page_{bond_url.split("/")[-1]}.html', 'w', encoding='utf-8') as f:
                    f.write(page_source)
                print(f"  已保存页面HTML到 debug_page_{bond_url.split('/')[-1]}.html")
            except:
                pass
        
        # 从字典中提取所需信息（不区分大小写）
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
        
        # Coupon - 可能包含%符号，需要提取数字
        for key in ['coupon', 'coupon rate', 'interest rate']:
            if key in data_dict_lower:
                coupon_text = data_dict_lower[key].strip()
                # 移除%符号并提取数字
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
        
        # Exchange - 可能在"Exchange"或"Traded on"字段
        for key in ['exchange', 'traded on', 'market', 'trading venue']:
            if key in data_dict_lower:
                bond_info['exchange'] = data_dict_lower[key].strip()
                break
        
        # Issue Price
        for key in ['issue price', 'price', 'issue']:
            if key in data_dict_lower:
                issue_price_text = data_dict_lower[key].strip()
                # 提取数字
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
        print(f"获取债券详情时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return bond_info

def extract_from_text(page_text, bond_info, bond_url):
    """从页面文本中提取信息（备选方案）"""
    # 使用正则表达式提取信息
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
    从债券页面提取tkData标识符
    tkData用于调用Chart_GetChartData API获取历史价格
    
    参考其他同学的成功实现，tkData通常在页面的JavaScript代码中
    格式类似: tkData="1,130654501,1330,184" 或 Chart_GetChartData?tkData=1%2C130654501...
    """
    try:
        # 等待页面完全加载，特别是JavaScript执行完成
        time.sleep(2)  # 给JavaScript更多时间执行
        
        # 尝试等待图表相关元素加载（如果存在）
        try:
            # 等待可能的图表容器加载
            WebDriverWait(driver, 5).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
        except:
            pass
        
        # 获取页面源代码
        page_source = driver.page_source
        
        # 方法1: 从Chart_GetChartData URL中提取tkData（最常见）
        # 查找类似: Chart_GetChartData?instrumentType=Bond&tkData=1%2C130654501%2C1330%2C184
        chart_url_pattern = r'Chart_GetChartData[^"\']*tkData=([^"\'&]+)'
        match = re.search(chart_url_pattern, page_source, re.IGNORECASE)
        if match:
            tkdata_encoded = match.group(1)
            # URL解码
            tkdata = unquote(tkdata_encoded)
            print(f"  从Chart URL找到tkData: {tkdata[:50]}...")
            return tkdata
        
        # 方法2: 从JavaScript变量中提取（更宽松的模式）
        # 查找类似: var tkData = "1,130654501,1330,184" 或 tkData:"1,130654501,1330,184"
        # 或者: tkData: "1,130654501,1330,184" 或 "tkData":"1,130654501,1330,184"
        js_patterns = [
            r'tkData["\']?\s*[:=]\s*["\']([^"\']+)["\']',  # tkData:"..." 或 tkData="..."
            r'tkData["\']?\s*=\s*["\']([^"\']+)["\']',      # tkData="..."
            r'["\']tkData["\']\s*:\s*["\']([^"\']+)["\']', # "tkData":"..."
            r'tkData\s*[:=]\s*([0-9,]+)',                   # tkData: 1,130654501,1330,184 (无引号)
        ]
        
        for pattern in js_patterns:
            matches = re.finditer(pattern, page_source, re.IGNORECASE)
            for match in matches:
                tkdata = match.group(1)
                # 如果包含URL编码，解码
                if '%' in tkdata:
                    tkdata = unquote(tkdata)
                # 验证tkData格式（应该包含逗号分隔的数字）
                if re.match(r'^[\d,]+$', tkdata.replace(' ', '')):
                    print(f"  从JavaScript变量找到tkData: {tkdata[:50]}...")
                    return tkdata
        
        # 方法3: 从script标签中提取（更仔细的搜索）
        try:
            scripts = driver.find_elements(By.TAG_NAME, "script")
            for script in scripts:
                script_text = script.get_attribute('innerHTML') or script.get_attribute('textContent') or ''
                if not script_text:
                    continue
                
                # 尝试多种模式
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
                        # 验证格式
                        if re.match(r'^[\d,]+$', tkdata.replace(' ', '')):
                            print(f"  从script标签找到tkData: {tkdata[:50]}...")
                            return tkdata
        except Exception as e:
            pass
        
        # 方法4: 从网络请求日志中提取（如果启用性能日志）
        # 这需要特殊配置，暂时跳过
        
        print("  警告: 无法从页面提取tkData")
        print("  提示: 可能需要检查页面是否完全加载，或页面结构是否不同")
        return None
        
    except Exception as e:
        print(f"  提取tkData时出错: {e}")
        return None

def fetch_historical_prices_via_api(bond_url, tkdata, dates, max_retries=3):
    """
    使用Chart_GetChartData API获取历史价格数据
    这是更可靠的方法，参考了其他同学的成功实现
    """
    import requests
    from urllib.parse import quote
    
    prices = {}
    
    if not tkdata:
        print("  无法获取历史价格：缺少tkData")
        return prices
    
    try:
        # 构建API URL
        # 日期格式：YYYYMMDD
        from_date = dates[0].replace('-', '') if dates else '20260105'
        to_date = dates[-1].replace('-', '') if dates else '20260119'
        
        # URL编码tkData
        tkdata_encoded = quote(tkdata, safe='')
        
        api_url = (
            f"https://markets.businessinsider.com/Ajax/Chart_GetChartData"
            f"?instrumentType=Bond&tkData={tkdata_encoded}&from={from_date}&to={to_date}"
        )
        
        print(f"  调用API: {api_url[:100]}...")
        
        # 发送请求（带重试机制和更长的超时时间）
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
                # 增加延迟，避免请求过快
                if attempt > 0:
                    wait_time = (attempt + 1) * 3  # 3, 6, 9秒
                    print(f"  等待{wait_time}秒后重试 (尝试 {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                
                response = requests.get(api_url, headers=headers, timeout=60)
                response.raise_for_status()
                break
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    continue  # 继续重试循环
                else:
                    print(f"  API请求超时，已重试{max_retries}次，放弃")
                    return prices
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 503:
                    # 503错误通常是服务器过载，需要更长的等待
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # 5, 10, 15秒
                        print(f"  服务器503错误，{wait_time}秒后重试 (尝试 {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  服务器503错误，已重试{max_retries}次，放弃")
                        return prices
                else:
                    # 其他HTTP错误，直接返回
                    print(f"  HTTP错误: {e}")
                    return prices
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    continue  # 继续重试
                else:
                    print(f"  API请求失败: {e}")
                    return prices
        
        if response is None:
            return prices
        
        # 解析JSON响应
        data = response.json()
        
        # 递归提取时间序列点（参考其他同学的实现思路）
        def extract_time_series_points(obj, points=None):
            """递归提取时间序列点 [timestamp, value]"""
            if points is None:
                points = []
            
            if isinstance(obj, list):
                # 如果是列表，检查是否是 [timestamp, value] 格式
                if len(obj) >= 2 and all(isinstance(x, (int, float)) for x in obj[:2]):
                    points.append((obj[0], obj[1]))
                else:
                    # 递归处理列表中的每个元素
                    for item in obj:
                        extract_time_series_points(item, points)
            elif isinstance(obj, dict):
                # 如果是字典，查找时间戳和值字段
                keys_lower = {str(k).lower(): k for k in obj.keys()}
                
                # 查找时间戳字段
                time_key = None
                for tkey in ['x', 't', 'time', 'date', 'timestamp']:
                    if tkey in keys_lower:
                        time_key = keys_lower[tkey]
                        break
                
                # 查找值字段
                value_key = None
                for vkey in ['y', 'v', 'value', 'close', 'price', 'last']:
                    if vkey in keys_lower:
                        value_key = keys_lower[vkey]
                        break
                
                # 如果找到时间和值字段，提取点
                if time_key and value_key:
                    points.append((obj[time_key], obj[value_key]))
                else:
                    # 递归处理字典中的每个值
                    for value in obj.values():
                        extract_time_series_points(value, points)
            
            return points
        
        # 提取所有时间序列点
        time_series_points = extract_time_series_points(data)
        
        if not time_series_points:
            print("  警告: API返回的数据中没有找到时间序列点")
            return prices
        
        print(f"  从API提取到 {len(time_series_points)} 个数据点")
        
        # 转换时间戳为日期，并提取价格
        for timestamp, value in time_series_points:
            try:
                # 转换时间戳为日期
                date_obj = None
                
                if isinstance(timestamp, str):
                    # 尝试解析字符串日期
                    try:
                        date_obj = datetime.strptime(timestamp, '%Y-%m-%d')
                    except:
                        try:
                            date_obj = pd.to_datetime(timestamp)
                        except:
                            continue
                elif isinstance(timestamp, (int, float)):
                    # 时间戳可能是毫秒或秒
                    if timestamp > 1e11:  # 毫秒
                        date_obj = datetime.fromtimestamp(timestamp / 1000)
                    else:  # 秒
                        date_obj = datetime.fromtimestamp(timestamp)
                else:
                    continue
                
                if date_obj is None:
                    continue
                
                date_str = date_obj.strftime('%Y-%m-%d')
                
                # 提取价格值
                price = float(value)
                
                # 验证价格合理性（债券价格通常在90-110之间，以面值百分比表示）
                if not (50 < price < 150):
                    continue
                
                # 只保存目标日期范围内的价格
                target_start = datetime.strptime(dates[0], '%Y-%m-%d').date()
                target_end = datetime.strptime(dates[-1], '%Y-%m-%d').date()
                date_only = date_obj.date()
                
                if date_str in dates or (target_start <= date_only <= target_end):
                    prices[date_str] = price
                    
            except (ValueError, TypeError, OSError) as e:
                continue
        
        print(f"  成功提取 {len(prices)} 个价格数据点")
        
    except requests.RequestException as e:
        print(f"  API请求失败: {e}")
    except json.JSONDecodeError as e:
        print(f"  JSON解析失败: {e}")
    except Exception as e:
        print(f"  获取历史价格时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return prices

def fetch_historical_prices(driver, bond_url, dates):
    """
    获取债券的历史价格数据
    使用API方法（更可靠）
    """
    print(f"正在获取历史价格: {bond_url}")
    
    prices = {}
    
    try:
        # 确保我们在债券详情页面
        if driver.current_url != bond_url:
            driver.get(bond_url)
            time.sleep(4)  # 增加等待时间，确保页面完全加载（包括JavaScript）
        
        # 从页面提取tkData（可能需要多次尝试）
        tkdata = None
        max_attempts = 2
        
        for attempt in range(max_attempts):
            tkdata = extract_tkdata_from_page(driver, bond_url)
            if tkdata:
                break
            elif attempt < max_attempts - 1:
                print(f"  第{attempt + 1}次尝试失败，等待后重试...")
                time.sleep(2)
                # 刷新页面
                driver.refresh()
                time.sleep(3)
        
        if tkdata:
            # 使用API方法获取价格
            prices = fetch_historical_prices_via_api(bond_url, tkdata, dates)
        else:
            print("  警告: 无法提取tkData，跳过此债券的价格数据")
            # 返回空字典，但不会导致整个流程失败
        
    except Exception as e:
        print(f"获取历史价格时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return prices

def normalize_date(date_str):
    """标准化日期格式为 YYYY-MM-DD"""
    # 尝试多种日期格式
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
    
    return date_str  # 如果无法解析，返回原字符串

def filter_bonds_by_maturity(bonds_data, max_years=10):
    """过滤债券：只保留到期日少于10年的债券（从2026年1月5日起）"""
    filtered = []
    reference_date = datetime(2026, 1, 5)
    
    for bond in bonds_data:
        maturity_date = bond.get('maturity_date')
        if not maturity_date:
            continue
        
        try:
            # 尝试解析到期日期
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
    """收集所有数据的主函数"""
    print("=" * 60)
    print("开始收集加拿大政府债券数据...")
    print("=" * 60)
    
    # 检查日期范围
    today = datetime.now()
    if START_DATE > today:
        print(f"警告: 开始日期 {START_DATE.strftime('%Y-%m-%d')} 在未来")
        print("如果数据不存在，可能需要使用历史数据或等待到指定日期")
        print()
    
    # 获取工作日列表
    weekdays = get_weekdays(START_DATE, END_DATE)
    print(f"需要收集 {len(weekdays)} 个工作日的价格数据")
    print(f"日期范围: {weekdays[0]} 到 {weekdays[-1]}")
    print()
    
    # 设置WebDriver
    driver = setup_driver(headless=headless)
    if not driver:
        print("无法启动WebDriver，请检查Chrome和ChromeDriver安装")
        return []
    
    try:
        all_bonds = []
        
        # 从两个URL获取债券列表
        for url in [SHORT_TERM_URL, MID_TERM_URL]:
            bonds = fetch_bond_list(driver, url)
            all_bonds.extend(bonds)
            time.sleep(2)
        
        # 去重（基于URL）
        seen_urls = set()
        unique_bonds = []
        for bond in all_bonds:
            if bond['url'] not in seen_urls:
                seen_urls.add(bond['url'])
                unique_bonds.append(bond)
        
        print(f"\n总共找到 {len(unique_bonds)} 个唯一债券")
        
        # 收集每个债券的详细信息
        bonds_data = []
        for i, bond in enumerate(unique_bonds, 1):
            print(f"\n处理债券 {i}/{len(unique_bonds)}: {bond.get('name', 'Unknown')}")
            
            # 获取债券详情
            details = fetch_bond_details(driver, bond['url'])
            if details:
                # 获取历史价格
                prices = fetch_historical_prices(driver, bond['url'], weekdays)
                details['historical_prices'] = prices
                details['price_count'] = len(prices)  # 添加价格数量用于调试
                bonds_data.append(details)
                
                # 打印提取到的数据（用于调试）
                print(f"  提取到的数据: coupon={details.get('coupon')}, isin={details.get('isin')}, "
                      f"issue_date={details.get('issue_date')}, maturity_date={details.get('maturity_date')}, "
                      f"exchange={details.get('exchange')}, 价格数量={len(prices)}")
            
            time.sleep(3)  # 增加延迟，避免请求过快和API限流（503错误通常是因为请求太频繁）
        
        # 暂时不过滤，保存所有原始数据用于调试
        print(f"\n收集到 {len(bonds_data)} 个债券的原始数据（未过滤）")
        
        # 保存原始数据（不过滤）
        output_file = 'bonds_data_raw.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(bonds_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n原始数据已保存到 {output_file}")
        
        # 也保存过滤后的数据（用于对比）
        filtered_bonds = filter_bonds_by_maturity(bonds_data, max_years=10)
        print(f"过滤后剩余 {len(filtered_bonds)} 个债券（到期日<10年）")
        
        # 保存过滤后的数据
        output_file_filtered = 'bonds_data.json'
        with open(output_file_filtered, 'w', encoding='utf-8') as f:
            json.dump(filtered_bonds, f, indent=2, ensure_ascii=False)
        
        print(f"过滤后的数据已保存到 {output_file_filtered}")
        
        # 使用原始数据生成CSV（包含所有债券）
        bonds_data_for_csv = bonds_data
        
        # 也保存为CSV格式（便于分析）- 使用原始数据
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
                # 即使没有价格数据，也保存债券基本信息
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
            print(f"原始数据CSV已保存到 {csv_file}")
        
        return bonds_data_for_csv
        
    finally:
        driver.quit()
        print("\nWebDriver已关闭")

if __name__ == "__main__":
    # 检查是否使用无头模式（默认True）
    headless = True
    if len(sys.argv) > 1 and sys.argv[1] == '--no-headless':
        headless = False
        print("使用有界面模式（便于调试）")
    
    data = collect_all_data(headless=headless)
    print(f"\n完成！共收集 {len(data)} 个债券的数据")
