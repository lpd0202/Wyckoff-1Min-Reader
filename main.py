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
import json
import random
import re
from typing import Optional

# ==========================================
# Gemini 稳定性增强：429 退避重试（保留，可作为备用）
# ==========================================
class GeminiQuotaExceeded(Exception):
    """按天/按项目配额耗尽：等待无效，应切 OpenAI。"""
    pass
class GeminiRateLimited(Exception):
    """短期速率限制：可退避重试。"""
    pass
def _extract_retry_seconds(resp: requests.Response) -> int:
    ra = resp.headers.get("Retry-After")
    if ra:
        try:
            return max(1, int(float(ra)))
        except:
            pass
    text = resp.text or ""
    m = re.search(r"retry in\s+([\d\.]+)\s*s", text, re.IGNORECASE)
    if m:
        return max(1, int(float(m.group(1))))
    try:
        obj = resp.json()
        msg = ((obj.get("error", {}) or {}).get("message", "") or "")
        m2 = re.search(r"retry in\s+([\d\.]+)\s*s", msg, re.IGNORECASE)
        if m2:
            return max(1, int(float(m2.group(1))))
    except:
        pass
    return 0
def _is_quota_exhausted(resp: requests.Response) -> bool:
    text = (resp.text or "").lower()
    if ("quota exceeded" in text) or ("exceeded your current quota" in text):
        return True
    if ("free_tier" in text) and ("limit" in text):
        return True
    try:
        obj = resp.json()
        msg = (((obj.get("error", {}) or {}).get("message", "")) or "").lower()
        if ("quota exceeded" in msg) or ("exceeded your current quota" in msg):
            return True
    except:
        pass
    return False
def call_gemini_http(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY missing")
    model_name = os.getenv("GEMINI_MODEL") or "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    session = requests.Session()
    headers = {"Content-Type": "application/json"}
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "system_instruction": {"parts": [{"text": "You are Richard D. Wyckoff."}]},
        "generationConfig": {"temperature": 0.2},
        "safetySettings": safety_settings,
    }
    max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "8"))
    base_sleep = float(os.getenv("GEMINI_BASE_SLEEP", "2.5"))
    timeout_s = int(os.getenv("GEMINI_TIMEOUT", "300"))
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(url, headers=headers, json=data, timeout=timeout_s)
            if resp.status_code == 200:
                result = resp.json()
                candidates = result.get("candidates", []) or []
                if not candidates:
                    raise ValueError(f"No candidates. Raw={str(result)[:400]}")
                content = candidates[0].get("content", {}) or {}
                parts = content.get("parts", []) or []
                if not parts:
                    raise ValueError(f"Empty parts. Raw={str(result)[:400]}")
                text = parts[0].get("text", "") or ""
                if not text:
                    raise ValueError(f"Empty text. Raw={str(result)[:400]}")
                return text
            if resp.status_code == 429:
                if _is_quota_exhausted(resp):
                    raise GeminiQuotaExceeded(resp.text[:1200])
                retry_s = _extract_retry_seconds(resp)
                if retry_s <= 0:
                    retry_s = int(base_sleep * (2 ** (attempt - 1)) + random.random() * 2)
                if attempt == max_retries:
                    raise GeminiRateLimited(resp.text[:1200])
                print(f"   ⚠️ Gemini 429(短期限流)，等待 {retry_s}s 后重试 ({attempt}/{max_retries})", flush=True)
                time.sleep(retry_s)
                continue
            if resp.status_code == 503:
                retry_s = int(base_sleep * (2 ** (attempt - 1)) + random.random() * 2)
                if attempt == max_retries:
                    raise Exception(f"Gemini 503 final: {resp.text[:1200]}")
                print(f"   ⚠️ Gemini 503(过载)，等待 {retry_s}s 后重试 ({attempt}/{max_retries})", flush=True)
                time.sleep(retry_s)
                continue
            raise Exception(f"Gemini HTTP {resp.status_code}: {resp.text[:1200]}")
        except GeminiQuotaExceeded:
            raise
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                raise
            retry_s = int(base_sleep * (2 ** (attempt - 1)) + random.random() * 2)
            print(f"   ⚠️ Gemini 调用异常：{str(e)[:200]}... 等待 {retry_s}s 重试 ({attempt}/{max_retries})", flush=True)
            time.sleep(retry_s)
    raise last_err or Exception("Gemini unknown failure")

# ==========================================
# 新增：硅基流动DeepSeek调用函数
# ==========================================
def call_deepseek_siliconflow(prompt: str) -> str:
    """调用硅基流动的DeepSeek API"""
    # 从环境变量获取密钥（已在GitHub Secrets中配置）
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY missing (请在GitHub Secrets中配置)")
    
    # 硅基流动接口地址
    url = "https://api.siliconflow.cn/v1/chat/completions"
    # 请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # 请求数据（威科夫分析系统角色）
    data = {
        "model": "DeepSeek-Coder-V2",  # 硅基流动支持的DeepSeek模型
        "messages": [
            {"role": "system", "content": "You are Richard D. Wyckoff. 请基于威科夫理论，分析股票的量价数据，识别Spring、UT、LPS等关键结构，结合用户持仓成本提供Hold/Sell/Stop-Loss建议，分析需专业、简洁，避免无关内容。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 2048
    }
    
    # 重试逻辑
    session = requests.Session()
    max_retries = 3
    base_sleep = 3
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(url, headers=headers, json=data, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            choices = result.get("choices", [])
            if not choices:
                raise ValueError(f"DeepSeek返回空结果: {str(result)[:400]}")
            content = choices[0].get("message", {}).get("content", "").strip()
            if not content:
                raise ValueError(f"DeepSeek返回空内容: {str(result)[:400]}")
            return content
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt == max_retries:
                raise Exception(f"DeepSeek调用失败({attempt}/{max_retries}): {str(e)[:200]}")
            retry_s = base_sleep * (2 ** (attempt - 1))
            print(f"   ⚠️ DeepSeek调用异常，等待 {retry_s}s 后重试 ({attempt}/{max_retries})", flush=True)
            time.sleep(retry_s)
    raise last_err or Exception("DeepSeek未知错误")

# ==========================================
# 数据获取模块（无修改）
# ==========================================
def fetch_stock_data_dynamic(symbol: str, buy_date_str: str) -> dict:
    clean_digits = ''.join(filter(str.isdigit, str(symbol)))
    symbol_code = clean_digits.zfill(6)
    start_date_em = (datetime.now() - timedelta(days=40)).strftime("%Y%m%d")
    try:
        df = ak.stock_zh_a_hist_min_em(symbol=symbol_code, period="5", start_date=start_date_em, adjust="qfq")
    except Exception as e:
        print(f"   [Error] {symbol_code} AkShare接口报错: {e}", flush=True)
        return {"df": pd.DataFrame(), "period": "5m"}
    if df.empty:
        return {"df": pd.DataFrame(), "period": "5m"}
    rename_map = {"时间": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    cols = ["open", "high", "low", "close", "volume"]
    valid_cols = [c for c in cols if c in df.columns]
    df[valid_cols] = df[valid_cols].astype(float)
    if "open" in df.columns and (df["open"] == 0).any():
        df["open"] = df["open"].replace(0, np.nan)
        if "close" in df.columns:
            df["open"] = df["open"].fillna(df["close"].shift(1)).fillna(df["close"])
    if len(df) > 500:
        df = df.tail(500).reset_index(drop=True)
    return {"df": df, "period": "5m"}
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "close" in df.columns:
        df["ma50"] = df["close"].rolling(50).mean()
        df["ma200"] = df["close"].rolling(200).mean()
    return df

# ==========================================
# 绘图模块（无修改）
# ==========================================
def generate_local_chart(symbol: str, df: pd.DataFrame, save_path: str, period: str):
    if df.empty:
        return
    plot_df = df.copy()
    if "date" in plot_df.columns:
        plot_df.set_index("date", inplace=True)
    mc = mpf.make_marketcolors(
        up='#ff3333',
        down='#00b060',
        edge='inherit',
        wick='inherit',
        volume={'up': '#ff3333', 'down': '#00b060'},
        inherit=True
    )
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc, gridstyle=':', y_on_right=True)
    apds = []
    if 'ma50' in plot_df.columns:
        apds.append(mpf.make_addplot(plot_df['ma50'], color='#ff9900', width=1.5))
    if 'ma200' in plot_df.columns:
        apds.append(mpf.make_addplot(plot_df['ma200'], color='#2196f3', width=2.0))
    try:
        mpf.plot(
            plot_df,
            type='candle',
            style=s,
            addplot=apds,
            volume=True,
            title=f"Wyckoff: {symbol} ({period} | {len(plot_df)} bars)",
            savefig=dict(fname=save_path, dpi=150, bbox_inches='tight'),
            warn_too_much_data=2000
        )
    except Exception as e:
        print(f"   [Error] {symbol} 绘图失败: {e}", flush=True)

# ==========================================
# AI 分析模块（修改：优先调用DeepSeek）
# ==========================================
_PROMPT_CACHE = None
def get_prompt_content(symbol, df, position_info):
    global _PROMPT_CACHE
    if _PROMPT_CACHE is None:
        prompt_template = os.getenv("WYCKOFF_PROMPT_TEMPLATE")
        if not prompt_template and os.path.exists("prompt_secret.txt"):
            try:
                with open("prompt_secret.txt", "r", encoding="utf-8") as f:
                    prompt_template = f.read()
            except:
                prompt_template = None
        _PROMPT_CACHE = prompt_template
    prompt_template = _PROMPT_CACHE
    if not prompt_template:
        return None
    csv_data = df.to_csv(index=False)
    latest = df.iloc[-1]
    base_prompt = (
        prompt_template.replace("{symbol}", symbol)
        .replace("{latest_time}", str(latest["date"]))
        .replace("{latest_price}", str(latest["close"]))
        .replace("{csv_data}", csv_data)
    )
    def safe_get(key):
        val = position_info.get(key)
        if val is None or str(val).lower() == 'nan' or str(val).strip() == '':
            return 'N/A'
        return val
    buy_date = safe_get('date')
    buy_price = safe_get('price')
    qty = safe_get('qty')
    position_text = (
        f"\n\n[USER POSITION DATA]\n"
        f"Symbol: {symbol}\n"
        f"Buy Date: {buy_date}\n"
        f"Cost Price: {buy_price}\n"
        f"Quantity: {qty}\n"
        f"(Note: Please analyze the current trend based on this position data. If position data is N/A, analyze as a potential new entry.)"
    )
    return base_prompt + position_text
def call_openai_official(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI Key missing")
    model_name = os.getenv("AI_MODEL", "gpt-4o")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are Richard D. Wyckoff."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return resp.choices[0].message.content
def ai_analyze(symbol, df, position_info):
    prompt = get_prompt_content(symbol, df, position_info)
    if not prompt:
        return "Error: No Prompt"
    # 优先调用硅基流动DeepSeek
    try:
        print(f"   🧠 正在调用 DeepSeek（硅基流动）进行威科夫分析...", flush=True)
        return call_deepseek_siliconflow(prompt)
    except Exception as e1:
        print(f"   ⚠️ DeepSeek调用失败，尝试切换到Gemini: {str(e1)[:150]}...", flush=True)
        # 兜底逻辑（保留原Gemini/OpenAI）
        try:
            return call_gemini_http(prompt)
        except GeminiQuotaExceeded as qe:
            print(f"   ⚠️ [{symbol}] Gemini 配额耗尽，切 OpenAI: {str(qe)[:160]}...", flush=True)
            try:
                return call_openai_official(prompt)
            except Exception as e2:
                return f"Analysis Failed. DeepSeek Error: {e1}. Gemini Quota Error: {qe}. OpenAI Error: {e2}"
        except GeminiRateLimited as rl:
            print(f"   ⚠️ [{symbol}] Gemini 短期限流，切 OpenAI: {str(rl)[:160]}...", flush=True)
            try:
                return call_openai_official(prompt)
            except Exception as e2:
                return f"Analysis Failed. DeepSeek Error: {e1}. Gemini RateLimit Error: {rl}. OpenAI Error: {e2}"
        except Exception as e2:
            print(f"   ⚠️ [{symbol}] Gemini 失败，切 OpenAI: {str(e2)[:160]}...", flush=True)
            try:
                return call_openai_official(prompt)
            except Exception as e3:
                return f"Analysis Failed. DeepSeek Error: {e1}. Gemini Error: {e2}. OpenAI Error: {e3}"

# ==========================================
# PDF 生成模块（无修改）
# ==========================================
def generate_pdf_report(symbol, chart_path, report_text, pdf_path):
    html_content = markdown.markdown(report_text)
    abs_chart_path = os.path.abspath(chart_path)
    font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    if not os.path.exists(font_path):
        font_path = "msyh.ttc"
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
    except:
        return False

# ==========================================
# 主程序（无修改）
# ==========================================
def process_one_stock(symbol: str, position_info: dict):
    if position_info is None:
        position_info = {}
    clean_digits = ''.join(filter(str.isdigit, str(symbol)))
    clean_symbol = clean_digits.zfill(6)
    print(f"🚀 [{clean_symbol}] 开始分析...", flush=True)
    data_res = fetch_stock_data_dynamic(clean_symbol, position_info.get('date'))
    df = data_res["df"]
    period = data_res["period"]
    if df.empty:
        print(f"   ⚠️ [{clean_symbol}] 数据为空，跳过", flush=True)
        return None
    df = add_indicators(df)
    beijing_tz = timezone(timedelta(hours=8))
    ts = datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S")
    csv_path = f"data/{clean_symbol}_{period}_{ts}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    chart_path = f"reports/{clean_symbol}_chart_{ts}.png"
    pdf_path = f"reports/{clean_symbol}_report_{period}_{ts}.pdf"
    generate_local_chart(clean_symbol, df, chart_path, period)
    report_text = ai_analyze(clean_symbol, df, position_info)
    if generate_pdf_report(clean_symbol, chart_path, report_text, pdf_path):
        print(f"✅ [{clean_symbol}] 报告生成完毕", flush=True)
        return pdf_path
    return None
def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    print("☁️ 正在连接 Google Sheets...", flush=True)
    try:
        sm = SheetManager()
        stocks_dict = sm.get_all_stocks()
        print(f"📋 获取 {len(stocks_dict)} 个任务", flush=True)
    except Exception as e:
        print(f"❌ Sheet 连接失败: {e}", flush=True)
        return
    generated_pdfs = []
    items = list(stocks_dict.items())
    for i, (symbol, info) in enumerate(items):
        try:
            pdf_path = process_one_stock(symbol, info)
            if pdf_path:
                generated_pdfs.append(pdf_path)
        except Exception as e:
            print(f"❌ [{symbol}] 处理发生异常: {e}", flush=True)
        if i < len(items) - 1:
            print("⏳ 强制冷却 60秒 (防止 API 429)...", flush=True)
            time.sleep(60)
    if generated_pdfs:
        print(f"\n📝 生成推送清单 ({len(generated_pdfs)}):", flush=True)
        with open("push_list.txt", "w", encoding="utf-8") as f:
            for pdf in generated_pdfs:
                print(f"   -> {pdf}")
                f.write(f"{pdf}\n")
    else:
        print("\n⚠️ 无报告生成", flush=True)
if __name__ == "__main__":
    main()
