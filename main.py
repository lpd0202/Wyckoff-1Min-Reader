import os
import time
import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
import akshare as ak
import mplfinance as mpf
from openai import OpenAI
import numpy as np
import markdown
from xhtml2pdf import pisa
from sheet_manager import SheetManager 

# ==========================================
# 1. 数据获取模块 (智能策略版)
# ==========================================

def fetch_stock_data_dynamic(symbol: str, buy_date_str: str) -> dict:
    """
    智能获取数据策略：
    1. 强制补全代码 (2641 -> 002641)
    2. 计算 start_date = buy_date - 15天
    3. 优先拉取 5分钟 K线，若过长则切换 15分钟
    """
    # === 核心修复: 确保传入 AkShare 的是 6 位数字字符串 ===
    raw_code = ''.join(filter(str.isdigit, str(symbol)))
    symbol_code = raw_code.zfill(6)
    
    print(f"   -> 正在分析 {symbol_code} (买入日期: {buy_date_str})...")

    # 1. 计算开始时间 (近似倒推15个自然日)
    try:
        if buy_date_str and buy_date_str != 'Unknown' and len(str(buy_date_str)) >= 10:
            # 兼容 Google Sheets 可能传来的 '2025-01-01' 或其他格式
            buy_dt = datetime.strptime(str(buy_date_str)[:10], "%Y-%m-%d")
            start_dt = buy_dt - timedelta(days=15) 
            start_date_em = start_dt.strftime("%Y%m%d")
        else:
            # 默认
            start_date_em = (datetime.now() - timedelta(days=15)).strftime("%Y%m%d")
    except Exception as e:
        print(f"   [Warn] 日期解析失败 ({buy_date_str}), 使用默认窗口: {e}")
        start_date_em = (datetime.now() - timedelta(days=15)).strftime("%Y%m%d")

    # 2. 尝试拉取 5分钟 K线
    try:
        df = ak.stock_zh_a_hist_min_em(
            symbol=symbol_code, 
            period="5", 
            start_date=start_date_em,
            adjust="qfq"
        )
    except Exception as e:
        print(f"   [Error] 5min接口报错: {e}")
        return {"df": pd.DataFrame(), "period": "5m"}

    if df.empty:
        return {"df": pd.DataFrame(), "period": "5m"}

    # 3. 策略判断: 数据是否过长
    current_period = "5m"
    if len(df) > 960:
        print(f"   [策略] 5分钟数据({len(df)}根)过长，切换至 15分钟 K线 (最近960根)...")
        try:
            # 15分钟线
            df_15 = ak.stock_zh_a_hist_min_em(symbol=symbol_code, period="15", adjust="qfq")
            # 统一列名
            rename_map = {"时间": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
            df_15 = df_15.rename(columns={k: v for k, v in rename_map.items() if k in df_15.columns})
            
            df = df_15.tail(960).reset_index(drop=True) 
            current_period = "15m"
        except Exception as e:
            print(f"   [Warn] 15min接口失败，回退5min截断: {e}")
            df = df.tail(960)

    # 4. 数据清洗
    rename_map = {
        "时间": "date", "开盘": "open", "最高": "high",
        "最低": "low", "收盘": "close", "成交量": "volume"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    cols = ["open", "high", "low", "close", "volume"]
    df[cols] = df[cols].astype(float)

    # 修复 Open=0
    if (df["open"] == 0).any():
        df["open"] = df["open"].replace(0, np.nan)
        df["open"] = df["open"].fillna(df["close"].shift(1))
        df["open"] = df["open"].fillna(df["close"])

    return {"df": df, "period": current_period}

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()
    return df

# ==========================================
# 2. 绘图模块
# ==========================================

def generate_local_chart(symbol: str, df: pd.DataFrame, save_path: str, period: str):
    if df.empty: return

    plot_df = df.copy()
    plot_df.set_index("date", inplace=True)

    mc = mpf.make_marketcolors(
        up='#ff3333', down='#00b060', 
        edge='inherit', wick='inherit', 
        volume={'up': '#ff3333', 'down': '#00b060'},
        inherit=True
    )
    s = mpf.make_mpf_style(
        base_mpf_style='yahoo', 
        marketcolors=mc, 
        gridstyle=':', 
        y_on_right=True
    )

    apds = []
    if 'ma50' in plot_df.columns:
        apds.append(mpf.make_addplot(plot_df['ma50'], color='#ff9900', width=1.5))
    if 'ma200' in plot_df.columns:
        apds.append(mpf.make_addplot(plot_df['ma200'], color='#2196f3', width=2.0))

    try:
        mpf.plot(
            plot_df, type='candle', style=s, addplot=apds, volume=True,
            title=f"Wyckoff Setup: {symbol} ({period})",
            savefig=dict(fname=save_path, dpi=150, bbox_inches='tight'),
            warn_too_much_data=2000
        )
    except Exception as e:
        print(f"   [Error] 绘图失败: {e}")

# ==========================================
# 3. AI 分析模块
# ==========================================

def get_prompt_content(symbol, df, position_info):
    prompt_template = os.getenv("WYCKOFF_PROMPT_TEMPLATE")
    if not prompt_template and os.path.exists("prompt_secret.txt"):
        try:
            with open("prompt_secret.txt", "r", encoding="utf-8") as f:
                prompt_template = f.read()
        except: pass
    if not prompt_template: return None

    csv_data = df.to_csv(index=False)
    latest = df.iloc[-1]
    current_price = float(latest["close"])
    
    # === 持仓盈亏注入 ===
    try:
        buy_price = float(position_info.get('price', 0))
        buy_date = position_info.get('date', 'Unknown')
    except:
        buy_price = 0
    
    position_context = ""
    if buy_price > 0:
        pnl_pct = ((current_price - buy_price) / buy_price) * 100
        sign = "+" if pnl_pct >= 0 else ""
        position_context = (
            f"\n\n[USER POSITION INFO]\n"
            f"- Buy Date: {buy_date}\n"
            f"- Buy Price: {buy_price}\n"
            f"- Current PnL: {sign}{pnl_pct:.2f}%\n"
            f"IMPORTANT: The user holds this position. Advice on Hold/Sell/Stop-Loss?"
        )
    else:
        position_context = "\n\n[USER POSITION INFO]\nNo open position. Advice on Buy/Wait?"

    final_prompt = prompt_template.replace("{symbol}", symbol) \
                          .replace("{latest_time}", str(latest["date"])) \
                          .replace("{latest_price}", str(latest["close"])) \
                          .replace("{csv_data}", csv_data)
    
    return final_prompt + position_context

def call_gemini_http(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: raise ValueError("GEMINI_API_KEY missing")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    print(f"   >>> Gemini ({model_name})...")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "system_instruction": {"parts": [{"text": "You are Richard D. Wyckoff."}]},
        "generationConfig": {"temperature": 0.2}
    }
    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code != 200: raise Exception(f"Gemini API Error: {resp.text}")
    try:
        return resp.json()['candidates'][0]['content']['parts'][0]['text']
    except:
        return f"Gemini Parsing Error. Raw: {resp.text}"

def call_openai_official(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: raise ValueError("OPENAI_API_KEY missing")
    model_name = os.getenv("AI_MODEL", "gpt-4o")
    print(f"   >>> OpenAI ({model_name})...")
    
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name, 
        messages=[{"role": "system", "content": "You are Richard D. Wyckoff."}, {"role": "user", "content": prompt}],
        temperature=0.2 
    )
    return resp.choices[0].message.content

def ai_analyze(symbol, df, position_info):
    prompt = get_prompt_content(symbol, df, position_info)
    if not prompt: return "Error: No Prompt"
    
    try: return call_gemini_http(prompt)
    except Exception as e: 
        print(f"   [Warn] Gemini 失败: {e} -> 切换 OpenAI")
        try: return call_openai_official(prompt)
        except Exception as e2: return f"Analysis Failed: {e2}"

# ==========================================
# 4. PDF 生成模块
# ==========================================

def generate_pdf_report(symbol, chart_path, report_text, pdf_path):
    html_content = markdown.markdown(report_text)
    abs_chart_path = os.path.abspath(chart_path)
    font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    if not os.path.exists(font_path): font_path = "msyh.ttc" 
    
    full_html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @font-face {{ font-family: "MyChineseFont"; src: url("{font_path}"); }}
            @page {{ size: A4; margin: 1cm; }}
            body {{ font-family: "MyChineseFont", sans-serif; font-size: 12px; line-height: 1.5; }}
            h1, h2, h3, p, div {{ font-family: "MyChineseFont", sans-serif; color: #2c3e50; }}
            img {{ width: 18cm; margin-bottom: 20px; }}
            .header {{ text-align: center; margin-bottom: 20px; color: #7f8c8d; font-size: 10px; }}
        </style>
    </head>
    <body>
        <div class="header">Wyckoff Quantitative Analysis | {symbol}</div>
        <img src="{abs_chart_path}" />
        <hr/>
        {html_content}
    </body>
    </html>
    """
    try:
        with open(pdf_path, "wb") as pdf_file:
            pisa.CreatePDF(full_html, dest=pdf_file)
        return True
    except Exception as e:
        print(f"   [Error] PDF 生成失败: {e}")
        return False

# ==========================================
# 5. 主程序
# ==========================================

def process_one_stock(symbol: str, position_info: dict, generated_files: list):
    print(f"\n{'='*40}\n🚀 开始分析: {symbol}\n{'='*40}")

    data_res = fetch_stock_data_dynamic(symbol, position_info.get('date'))
    df = data_res["df"]
    period = data_res["period"]
    
    if df.empty:
        print(f"   [Skip] 数据为空，跳过 {symbol}")
        return
    df = add_indicators(df)

    # 文件名生成
    beijing_tz = timezone(timedelta(hours=8))
    ts = datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S")
    
    # 强制补全 symbol 以防万一
    clean_symbol = str(symbol).strip().zfill(6)
    
    chart_path = f"reports/{clean_symbol}_chart_{ts}.png"
    pdf_path = f"reports/{clean_symbol}_report_{period}_{ts}.pdf"
    
    generate_local_chart(clean_symbol, df, chart_path, period)
    report_text = ai_analyze(clean_symbol, df, position_info)
    
    if generate_pdf_report(clean_symbol, chart_path, report_text, pdf_path):
        generated_files.append(pdf_path)
    
    print(f"✅ {clean_symbol} 处理完成")

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("☁️ 正在连接 Google Sheets...")
    try:
        sm = SheetManager()
        stocks_dict = sm.get_all_stocks()
        print(f"📋 成功获取 {len(stocks_dict)} 只股票任务")
    except Exception as e:
        print(f"❌ Google Sheets 连接失败: {e}")
        return

    if not stocks_dict:
        print("⚠️ 列表为空，结束。")
        return

    generated_pdfs = []
    
    for i, (symbol, info) in enumerate(stocks_dict.items()):
        try:
            process_one_stock(symbol, info, generated_pdfs)
        except Exception as e:
            print(f"❌ {symbol} 错误: {e}")
        
        if i < len(stocks_dict) - 1:
            time.sleep(5)

    if generated_pdfs:
        print(f"\n📝 生成推送清单 ({len(generated_pdfs)} 个文件):")
        with open("push_list.txt", "w", encoding="utf-8") as f:
            for pdf in generated_pdfs:
                print(f"   -> {pdf}")
                f.write(f"{pdf}\n")
    else:
        print("\n⚠️ 本次没有生成任何 PDF")

if __name__ == "__main__":
    main()
