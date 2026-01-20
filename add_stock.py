import os
import re
import requests
from sheet_manager import SheetManager

def get_telegram_updates(bot_token, offset=None):
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    params = {"timeout": 10}
    if offset:
        params["offset"] = offset
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("result", [])
    except Exception as e:
        print(f"   ⚠️ 获取 Telegram 消息失败: {e}")
    return []

def send_telegram_message(bot_token, chat_id, text):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown" # 开启 Markdown 以便支持等宽字体
    }
    try:
        requests.post(url, json=data, timeout=10)
    except:
        pass

def parse_command(text):
    text = text.strip()
    code_match = re.search(r"\d{6}", text)
    if not code_match: return None
    code = code_match.group()
    
    intent = "add"
    if any(k in text for k in ["删除", "移除", "del", "remove", "取消"]):
        intent = "remove"
    
    remain_text = text.replace(code, "").replace("关注", "").replace("add", "")
    
    date = ""
    date_match = re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", remain_text)
    if date_match:
        date = date_match.group()
        remain_text = remain_text.replace(date, "")
    
    nums = re.findall(r"\d+\.?\d*", remain_text)
    price = ""
    qty = ""
    if len(nums) >= 1: price = nums[0]
    if len(nums) >= 2: qty = nums[1]
    
    return {
        "intent": intent, "code": code, "date": date, "price": price, "qty": qty
    }

def main():
    bot_token = os.getenv("TG_BOT_TOKEN")
    if not bot_token:
        print("❌ 缺少 TG_BOT_TOKEN")
        return

    print("☁️ 正在连接 Google Sheets...")
    try:
        sm = SheetManager()
        print("✅ 表格连接成功")
    except Exception as e:
        print(f"❌ 表格连接失败: {e}")
        return

    updates = get_telegram_updates(bot_token)
    if not updates:
        print("📭 无新消息")
        return

    print(f"📥 收到 {len(updates)} 条消息，开始处理...")
    
    max_update_id = 0
    
    for update in updates:
        update_id = update["update_id"]
        if update_id > max_update_id:
            max_update_id = update_id
            
        message = update.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        text = message.get("text", "")
        
        if not text or not chat_id: continue
        
        print(f"  -- 处理消息: {text}")
        
        parsed = parse_command(text)
        if not parsed:
            print("     -> 忽略 (非指令)")
            continue
            
        # 1. 执行增删改操作
        action_result = ""
        if parsed["intent"] == "remove":
            action_result = sm.remove_stock(parsed["code"])
        else:
            try:
                action_result = sm.add_or_update_stock(
                    parsed["code"], parsed["date"], parsed["price"], parsed["qty"]
                )
            except Exception as e:
                action_result = f"❌ 操作失败: {e}"
        
        # 2. 【关键】无论成功失败，都拉取最新的全量持仓
        portfolio_summary = sm.get_portfolio_summary()
        
        # 3. 拼接最终回复
        final_reply = f"{action_result}\n{portfolio_summary}"
        
        print(f"     -> 结果已发送")
        send_telegram_message(bot_token, chat_id, final_reply)

    if max_update_id > 0:
        print(f"🧹 清理消息队列 (Offset: {max_update_id + 1})")
        get_telegram_updates(bot_token, offset=max_update_id + 1)

if __name__ == "__main__":
    main()
