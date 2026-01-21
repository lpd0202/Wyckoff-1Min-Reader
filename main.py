import os
import time
import json
import akshare as ak
import pandas as pd
import mplfinance as mpf
import requests
from datetime import datetime, timedelta
from telegram import Bot
from telegram.error import TelegramError
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import openai

# ====================== 全局配置 ======================
# 环境变量（建议通过 GitHub Secrets 配置）
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_SHEETS_CRED_JSON = os.getenv("GOOGLE_SHEETS_CRED_JSON")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")  # Google Sheets ID（优先）
SPREADSHEET_NAME = os.getenv("SPREADSHEET_NAME")  # 备选：表格文件名

# AI 配置
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent"
OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"
TIMEOUT_SECONDS = 120
WYCKOFF_PROMPT_TEMPLATE = """
请基于以下A股{stock_code}（{stock_name}）的{period}分钟K线数据，按照威科夫（Wyckoff）理论分析：
1. 识别是否存在Spring（弹簧效应）、UT（上冲回落）、LPS（最后支撑点）等关键行为；
2. 分析供求关系和主力资金动向（吸筹/派发）；
3. 结合持仓成本{cost_price}、持仓数量{hold_num}、买入日期{buy_date}，给出明确的操作建议（Hold/Sell/Stop-Loss）；
4. 输出格式要求：分点说明，逻辑清晰，结论明确。

K线数据：
{klines_data}
"""

# ====================== 工具函数 ======================
def init_google_sheets():
    """初始化Google Sheets连接"""
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            json.loads(GOOGLE_SHEETS_CRED_JSON),
            ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        )
        client = gspread.authorize(creds)
        if SPREADSHEET_ID:
            sheet = client.open_by_key(SPREADSHEET_ID).sheet1
        else:
            sheet = client.open(SPREADSHEET_NAME).sheet1
        return sheet
    except Exception as e:
        raise Exception(f"Google Sheets初始化失败: {str(e)}")

def get_stock_list_from_sheets():
    """从Google Sheets获取持仓列表"""
    sheet = init_google_sheets()
    data = sheet.get_all_records()
    # 数据清洗：补全股票代码6位、过滤空值
    stock_list = []
    for row in data:
        stock_code = str(row.get("股票代码", "")).zfill(6)
        if not stock_code or stock_code == "000000":
            continue
        stock_list.append({
            "code": stock_code,
            "name": row.get("股票名称", ""),
            "buy_date": row.get("买入日期", ""),
            "cost": row.get("持仓成本", 0.0),
            "num": row.get("持仓数量", 0)
        })
    return stock_list

def fetch_stock_data_dynamic(stock_code, buy_date=None):
    """智能获取K线数据（优先5分钟，补全代码，回溯时间窗口）"""
    # 代码归一化：强制补全6位
    stock_code = stock_code.zfill(6)
    try:
        # 计算回溯窗口：买入日期前15天（无则默认近30天）
        end_date = datetime.now().strftime("%Y%m%d")
        if buy_date:
            buy_dt = datetime.strptime(buy_date, "%Y-%m-%d")
            start_date = (buy_dt - timedelta(days=15)).strftime("%Y%m%d")
        else:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        
        # 优先获取5分钟K线
        df = ak.stock_zh_a_hist_min_em(
            symbol=stock_code,
            period="5",  # 5分钟级别
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        if df.empty:
            # 降级到1分钟K线
            df = ak.stock_zh_a_hist_min_em(
                symbol=stock_code,
                period="1",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
        
        # 数据格式化
        df.rename(columns={
            "时间": "datetime", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume"
        }, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        return df
    except Exception as e:
        raise Exception(f"获取{stock_code}K线数据失败: {str(e)}")

def generate_wyckoff_analysis(stock_info, kline_df):
    """双AI引擎分析威科夫结构"""
    # 构造Prompt
    prompt = WYCKOFF_PROMPT_TEMPLATE.format(
        stock_code=stock_info["code"],
        stock_name=stock_info["name"],
        period=kline_df.index.inferred_freq.split("T")[0] if kline_df.index.inferred_freq else "5",
        klines_data=kline_df.tail(100).to_string(),  # 取最近100根K线
        cost_price=stock_info["cost"],
        hold_num=stock_info["num"],
        buy_date=stock_info["buy_date"]
    )

    # 1. 尝试Gemini引擎
    try:
        gemini_headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }
        gemini_data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "safetySettings": [{"category": "HARM_CATEGORY_ALL", "threshold": "BLOCK_NONE"}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2000}
        }
        gemini_resp = requests.post(
            f"{GEMINI_BASE_URL}?key={GEMINI_API_KEY}",
            json=gemini_data,
            timeout=TIMEOUT_SECONDS
        )
        gemini_resp.raise_for_status()
        gemini_result = gemini_resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        if gemini_result.strip():
            return "【Gemini分析结果】\n" + gemini_result
    except Exception as e:
        print(f"Gemini分析失败: {str(e)}")

    # 2. 降级到GPT-4o
    try:
        openai.api_key = OPENAI_API_KEY
        gpt_resp = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            timeout=TIMEOUT_SECONDS
        )
        gpt_result = gpt_resp.choices[0].message["content"]
        return "【GPT-4o分析结果（Gemini降级）】\n" + gpt_result
    except Exception as e:
        raise Exception(f"双AI引擎均失败: {str(e)}")

def plot_kline(stock_code, kline_df, save_path="kline_chart.png"):
    """绘制高对比K线图"""
    # 红绿配色（适配威科夫分析视觉）
    mc = mpf.make_marketcolors(up="red", down="green", inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, figratio=(12, 8), figscale=1.2)
    
    # 绘制K线
    mpf.plot(
        kline_df.tail(50),  # 最近50根K线
        type="candle",
        style=s,
        title=f"{stock_code} 威科夫分析K线",
        ylabel="价格 (¥)",
        volume=True,
        savefig=save_path
    )
    return save_path

def send_telegram_message(content, image_path=None):
    """发送消息/图片到Telegram"""
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    try:
        # 发送文本
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=content, parse_mode="Markdown")
        # 发送图片（K线图）
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=f)
    except TelegramError as e:
        raise Exception(f"Telegram推送失败: {str(e)}")

# ====================== 主流程 ======================
def main():
    """主执行函数"""
    print(f"===== 威科夫分析任务启动 {datetime.now()} =====")
    try:
        # 1. 获取持仓列表
        stock_list = get_stock_list_from_sheets()
        if not stock_list:
            print("未从Google Sheets获取到持仓数据")
            send_telegram_message("⚠️ 未检测到持仓数据，任务终止")
            return

        # 2. 遍历分析每只股票
        for stock in stock_list:
            print(f"\n分析股票: {stock['code']} - {stock['name']}")
            # 获取K线数据
            kline_df = fetch_stock_data_dynamic(stock["code"], stock["buy_date"])
            if kline_df.empty:
                send_telegram_message(f"❌ {stock['code']} {stock['name']} 未获取到K线数据")
                continue
            
            # 生成威科夫分析
            analysis_result = generate_wyckoff_analysis(stock, kline_df)
            
            # 绘制K线图
            kline_path = f"{stock['code']}_kline.png"
            plot_kline(stock["code"], kline_df, kline_path)
            
            # 推送结果到Telegram
            msg = f"""
📈 【{stock['code']} {stock['name']} 威科夫分析报告】
📅 买入日期: {stock['buy_date'] or '无'}
💰 持仓成本: ¥{stock['cost']}
📊 分析结论:
{analysis_result}
            """
            send_telegram_message(msg, kline_path)
            
            # 清理临时文件
            if os.path.exists(kline_path):
                os.remove(kline_path)

        print(f"\n===== 任务完成 {datetime.now()} =====")
        send_telegram_message("✅ 所有持仓股票分析完成，报告已推送")

    except Exception as e:
        error_msg = f"❌ 任务执行失败: {str(e)}"
        print(error_msg)
        send_telegram_message(error_msg)

if __name__ == "__main__":
    main()
