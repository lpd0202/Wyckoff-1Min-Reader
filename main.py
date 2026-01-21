import os
import time
import json
import requests
import akshare as ak
import mplfinance as mpf
import pandas as pd
from datetime import datetime, timedelta
import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ===================== 全局配置 =====================
# 轨迹流动 API 配置
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V3.1-Terminus"

# Telegram 配置
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Google Sheets 配置
GOOGLE_CREDENTIALS = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")

# 全局参数
TIMEOUT = 120  # API 请求超时时间
STOCK_CODE_ZFILL = 6  # 股票代码补零位数
ANALYSIS_WINDOW_DAYS = 15  # 分析窗口天数

# ===================== 工具函数 =====================
def format_stock_code(stock_code: str) -> str:
    """补全股票代码为6位（处理Excel/Sheets丢零问题）"""
    return str(stock_code).zfill(STOCK_CODE_ZFILL)

def get_google_sheets_data() -> pd.DataFrame:
    """从Google Sheets读取持仓/关注列表"""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(GOOGLE_CREDENTIALS, scope)
    client = gspread.authorize(creds)
    
    # 连接表格（优先ID，兼容文件名）
    try:
        sheet = client.open_by_key(SPREADSHEET_ID).sheet1
    except:
        sheet = client.open(SPREADSHEET_ID).sheet1
    
    # 读取数据并转为DataFrame
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    # 补全股票代码
    if "股票代码" in df.columns:
        df["股票代码"] = df["股票代码"].apply(format_stock_code)
    return df

def fetch_stock_data_dynamic(stock_code: str, buy_date: str = None) -> pd.DataFrame:
    """
    智能获取股票K线数据
    :param stock_code: 6位股票代码
    :param buy_date: 买入日期（格式YYYY-MM-DD），为空则取最新数据
    :return: 标准化的K线DataFrame
    """
    # 计算分析窗口起始时间
    if buy_date:
        start_date = (datetime.strptime(buy_date, "%Y-%m-%d") - timedelta(days=ANALYSIS_WINDOW_DAYS)).strftime("%Y%m%d")
    else:
        start_date = (datetime.now() - timedelta(days=ANALYSIS_WINDOW_DAYS)).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")

    # 优先获取5分钟K线，兼容1分钟数据
    try:
        # AkShare获取A股5分钟K线
        stock_df = ak.stock_zh_a_hist_min_em(
            symbol=stock_code,
            period="5",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
    except Exception as e:
        # 降级获取1分钟K线
        stock_df = ak.stock_zh_a_hist_min_em(
            symbol=stock_code,
            period="1",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )

    # 数据标准化
    stock_df.rename(
        columns={
            "时间": "datetime",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume"
        },
        inplace=True
    )
    stock_df["datetime"] = pd.to_datetime(stock_df["datetime"])
    stock_df.set_index("datetime", inplace=True)
    return stock_df

def plot_kline(stock_df: pd.DataFrame, stock_code: str, save_path: str):
    """绘制高对比K线图并保存"""
    # 红绿配色（符合A股习惯）
    mc = mpf.make_marketcolors(up="red", down="green", inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, figratio=(12, 8), figscale=1.2)
    
    # 绘制K线
    mpf.plot(
        stock_df,
        type="candle",
        volume=True,
        style=s,
        title=f"{stock_code} Wyckoff 结构分析",
        ylabel="价格 (¥)",
        ylabel_lower="成交量",
        savefig=save_path
    )

def deepseek_ai_analysis(stock_data_str: str, position_info: str) -> str:
    """
    调用轨迹流动DeepSeek模型进行威科夫结构分析
    :param stock_data_str: 股票K线数据文本
    :param position_info: 持仓信息（成本/数量/买入日期）
    :return: AI分析结论
    """
    # 构建威科夫分析Prompt
    system_prompt = """
    你是专业的威科夫（Wyckoff）交易策略分析师，精通A股1分钟/5分钟微观结构分析。
    请基于提供的股票K线数据和持仓信息，完成以下分析：
    1. 识别供求关系变化，标注Spring（弹簧效应）、UT（上冲回落）、LPS（最后支撑点）等关键行为；
    2. 结合用户持仓成本/买入日期，给出明确的操作建议（Hold/Sell/Stop-Loss）及止损位；
    3. 分析过程需基于威科夫核心理论，拒绝情绪化、模糊化表述；
    4. 输出语言为中文，结构清晰，优先标注关键信号，再给出建议。
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"### 股票数据：\n{stock_data_str}\n### 持仓信息：\n{position_info}"}
    ]

    # 构造请求体
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.1,  # 低随机性保证分析稳定
        "max_tokens": 2000
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}"
    }

    # 发送请求
    try:
        response = requests.post(
            SILICONFLOW_API_URL,
            headers=headers,
            json=payload,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise Exception(f"DeepSeek API 调用失败: {str(e)}")

def generate_pdf_report(analysis_result: str, kline_img_path: str, report_path: str):
    """生成包含分析结论和K线图的PDF研报（简化版，可扩展ReportLab）"""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    
    # 设置字体
    pdf.add_font("SimHei", "", "SimHei.ttf", uni=True)
    pdf.set_font("SimHei", size=12)
    
    # 添加标题
    pdf.cell(200, 10, txt="Wyckoff-M1-Sentinel 量化分析报告", ln=True, align="C")
    pdf.ln(10)
    
    # 添加分析内容
    pdf.multi_cell(0, 10, txt=analysis_result)
    pdf.ln(5)
    
    # 添加K线图
    if os.path.exists(kline_img_path):
        pdf.image(kline_img_path, x=10, y=pdf.get_y(), w=180)
    
    # 保存PDF
    pdf.output(report_path)

async def send_telegram_message(content: str, file_path: str = None):
    """发送消息/文件到Telegram"""
    bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
    # 发送文本
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=content)
    # 发送PDF报告
    if file_path and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            await bot.send_document(chat_id=TELEGRAM_CHAT_ID, document=f)

# ===================== 核心业务逻辑 =====================
async def analyze_single_stock(stock_code: str, position_info: dict):
    """分析单只股票并生成报告"""
    try:
        # 1. 获取股票数据
        stock_df = fetch_stock_data_dynamic(
            stock_code=stock_code,
            buy_date=position_info.get("买入日期")
        )
        if stock_df.empty:
            await send_telegram_message(f"⚠️ {stock_code} 未获取到有效K线数据")
            return
        
        # 2. 绘制K线图
        kline_img_path = f"{stock_code}_kline.png"
        plot_kline(stock_df, stock_code, kline_img_path)
        
        # 3. 格式化数据供AI分析
        stock_data_str = stock_df.tail(100).to_string()  # 取最新100条数据
        position_info_str = json.dumps(position_info, ensure_ascii=False, indent=2)
        
        # 4. DeepSeek AI分析
        analysis_result = deepseek_ai_analysis(stock_data_str, position_info_str)
        
        # 5. 生成PDF报告
        report_path = f"{stock_code}_wyckoff_report.pdf"
        generate_pdf_report(analysis_result, kline_img_path, report_path)
        
        # 6. 推送至Telegram
        await send_telegram_message(
            content=f"✅ {stock_code} 威科夫分析完成：\n{analysis_result[:500]}...",
            file_path=report_path
        )
        
        # 清理临时文件
        for tmp_file in [kline_img_path, report_path]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
                
    except Exception as e:
        await send_telegram_message(f"❌ {stock_code} 分析失败：{str(e)}")

async def batch_analyze_stocks():
    """批量分析Google Sheets中的股票"""
    try:
        # 读取持仓列表
        stock_df = get_google_sheets_data()
        if stock_df.empty:
            await send_telegram_message("⚠️ Google Sheets 未读取到持仓数据")
            return
        
        # 遍历分析每只股票
        for _, row in stock_df.iterrows():
            position_info = {
                "股票代码": row.get("股票代码"),
                "买入日期": row.get("买入日期"),
                "持仓成本": row.get("持仓成本"),
                "持仓数量": row.get("持仓数量")
            }
            await analyze_single_stock(
                stock_code=position_info["股票代码"],
                position_info=position_info
            )
            # 避免API限流
            time.sleep(5)
            
    except Exception as e:
        await send_telegram_message(f"❌ 批量分析失败：{str(e)}")

# ===================== Telegram Bot 指令处理 =====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Telegram /start 指令"""
    await update.message.reply_text("📈 Wyckoff-M1-Sentinel 已启动\n指令列表：\n/analyze - 立即执行批量分析\n/refresh - 同步Google Sheets数据")

async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Telegram /analyze 指令：立即执行分析"""
    await update.message.reply_text("🔍 开始批量分析股票，请稍候...")
    await batch_analyze_stocks()
    await update.message.reply_text("✅ 批量分析完成，报告已推送！")

async def cmd_refresh(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Telegram /refresh 指令：同步Google Sheets数据"""
    try:
        stock_df = get_google_sheets_data()
        await update.message.reply_text(f"🔄 同步完成！当前监控股票数量：{len(stock_df)}")
    except Exception as e:
        await update.message.reply_text(f"❌ 同步失败：{str(e)}")

def run_telegram_bot():
    """启动Telegram Bot"""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # 注册指令处理器
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("analyze", cmd_analyze))
    application.add_handler(CommandHandler("refresh", cmd_refresh))
    
    # 启动Bot
    application.run_polling()

# ===================== 主入口 =====================
if __name__ == "__main__":
    # 优先级：1. 执行批量分析 2. 启动Telegram Bot
    mode = os.getenv("RUN_MODE", "batch")  # 通过环境变量控制运行模式
    if mode == "batch":
        # 非阻塞运行批量分析（适配GitHub Actions）
        import asyncio
        asyncio.run(batch_analyze_stocks())
    elif mode == "bot":
        # 启动Telegram Bot（持续运行）
        run_telegram_bot()
