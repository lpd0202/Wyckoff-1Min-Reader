import os
import json
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import traceback

class SheetManager:
    def __init__(self):
        print("   >>> [System] 初始化 Google Sheets (智能连接版)...")
        
        # 1. 读取环境变量
        json_str = os.getenv("GCP_SA_KEY")
        target_name = os.getenv("SHEET_NAME") # 您的值: "Wyckoff_Stock_List"
        
        if not json_str:
            raise ValueError("❌ 环境变量缺失: GCP_SA_KEY")
        if not target_name:
            raise ValueError("❌ 环境变量缺失: SHEET_NAME")

        # 2. 解析 JSON
        try:
            creds_dict = json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError("❌ GCP_SA_KEY 格式错误")

        # 3. 创建凭证
        try:
            SCOPES = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
            self.client = gspread.authorize(creds)
            print("   ✅ Google Auth 认证成功")
            # 打印机器人邮箱，方便您核对权限
            print(f"   🤖 当前机器人: {creds_dict.get('client_email')}")
            
        except Exception as e:
            raise Exception(f"认证环节崩溃: {str(e)}")

        # 4. 连接表格 (优先尝试名称，因为您明确说是用名称)
        self.sheet = None
        
        # 逻辑：先试着当文件名打开
        try:
            print(f"   >>> 正在尝试按【文件名】打开: '{target_name}'...")
            self.sheet = self.client.open(target_name).sheet1
            print("   ✅ [成功] 已通过文件名连接到表格！")
            
        except gspread.exceptions.SpreadsheetNotFound:
            # 如果找不到，再试一次是不是 ID (以防万一)
            print(f"   ⚠️ 按文件名未找到，尝试按 ID 打开...")
            try:
                self.sheet = self.client.open_by_key(target_name).sheet1
                print("   ✅ [成功] 原来这是一个 ID，连接成功！")
            except Exception:
                print(f"\n   ❌ [致命错误] 找不到表格: '{target_name}'")
                print(f"   请务必检查：")
                print(f"   1. 表格文件名是否完全一致 (注意空格)？")
                print(f"   2. 是否已点击 Share，并把机器人邮箱加为 Editor？")
                print(f"      (机器人邮箱见上方日志)")
                raise Exception("无法打开 Google Sheet")

    def get_all_stocks(self):
        """读取所有股票"""
        try:
            records = self.sheet.get_all_records()
            if not records:
                print("   ⚠️ 表格为空，无数据")
                return {}
            return self._parse_records(records)
        except Exception as e:
            print(f"   ⚠️ 读取数据失败: {e}")
            return {}

    def _parse_records(self, records):
        """解析数据辅助函数"""
        stocks = {}
        for row in records:
            # 1. 模糊匹配 'Code' 列
            code_key = next((k for k in row.keys() if 'Code' in str(k)), None)
            if not code_key: continue

            # 2. 强制补全 6 位代码 (2641 -> 002641)
            raw_val = row[code_key]
            clean_digits = ''.join(filter(str.isdigit, str(raw_val)))
            code = clean_digits.zfill(6)
            
            if not code or code == "000000": continue

            # 3. 读取其他字段
            date = str(row.get('BuyDate', '')).strip() or datetime.now().strftime("%Y-%m-%d")
            qty = str(row.get('Qty', '')).strip() or "0"
            price = str(row.get('Price', '')).strip() or "0.0"
            
            stocks[code] = {'date': date, 'qty': qty, 'price': price}
            
        return stocks

    def add_or_update_stock(self, code, date=None, qty=None, price=None):
        clean_digits = ''.join(filter(str.isdigit, str(code)))
        code = clean_digits.zfill(6)
        
        date = date or datetime.now().strftime("%Y-%m-%d")
        qty = qty or 0
        price = price or 0.0
        
        try:
            cell = self.sheet.find(code)
            self.sheet.update_cell(cell.row, 2, date)
            self.sheet.update_cell(cell.row, 3, qty)
            self.sheet.update_cell(cell.row, 4, price)
            return "Updated"
        except gspread.exceptions.CellNotFound:
            self.sheet.append_row([code, date, qty, price])
            return "Added"

    def remove_stock(self, code):
        clean_digits = ''.join(filter(str.isdigit, str(code)))
        code = clean_digits.zfill(6)
        try:
            cell = self.sheet.find(code)
            self.sheet.delete_rows(cell.row)
            return True
        except gspread.exceptions.CellNotFound:
            return False

    def clear_all(self):
        self.sheet.resize(rows=1) 
        self.sheet.resize(rows=100)
