import os
import json
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import traceback

class SheetManager:
    def __init__(self):
        print("   >>> [System] 初始化 Google Sheets (官方库认证版)...")
        
        # 1. 读取环境变量
        json_str = os.getenv("GCP_SA_KEY")
        sheet_key = os.getenv("SHEET_NAME")
        
        if not json_str:
            raise ValueError("❌ 环境变量缺失: GCP_SA_KEY")
        if not sheet_key:
            raise ValueError("❌ 环境变量缺失: SHEET_NAME")

        # 2. 解析 JSON
        try:
            creds_dict = json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError("❌ GCP_SA_KEY 格式错误 (JSON解析失败)")

        # 3. 创建凭证 (最稳健的方式)
        try:
            SCOPES = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
            self.client = gspread.authorize(creds)
            print("   ✅ Google Auth 认证成功")
            
        except Exception as e:
            raise Exception(f"认证环节崩溃: {str(e)}")

        # 4. 连接表格 (优先尝试通过 ID 打开)
        try:
            # 尝试把 SHEET_NAME 当作 ID 处理
            self.sheet = self.client.open_by_key(sheet_key).sheet1
            print(f"   ✅ 通过 ID 连接表格成功: {sheet_key[:6]}...")
            
        except gspread.exceptions.APIError:
            # 如果 ID 失败，可能是用户填的是文件名，尝试通过文件名打开
            print(f"   ⚠️ ID 连接失败，尝试通过名称打开: {sheet_key}...")
            try:
                self.sheet = self.client.open(sheet_key).sheet1
                print("   ✅ 通过名称连接表格成功")
            except Exception as e2:
                print(f"   ❌ 致命错误: 无法打开表格。请确认 GitHub Secret 'SHEET_NAME' 是正确的 表格ID (推荐) 或 文件名。")
                raise e2

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
            traceback.print_exc()
            return {}

    def _parse_records(self, records):
        """解析数据辅助函数"""
        stocks = {}
        for row in records:
            # 1. 模糊匹配 'Code' 列 (防止 Excel 里多打了空格)
            code_key = next((k for k in row.keys() if 'Code' in str(k)), None)
            if not code_key: continue

            # 2. === 核心修复: 强制补全 6 位代码 ===
            # 将 2641 变成 '002641'
            raw_val = row[code_key]
            code = str(raw_val).strip()
            
            # 过滤掉空行
            if not code: continue
            
            # 补零
            if code.isdigit():
                code = code.zfill(6)

            # 3. 读取其他字段
            date = str(row.get('BuyDate', '')).strip() or datetime.now().strftime("%Y-%m-%d")
            qty = str(row.get('Qty', '')).strip() or "0"
            price = str(row.get('Price', '')).strip() or "0.0"
            
            stocks[code] = {'date': date, 'qty': qty, 'price': price}
            
        return stocks

    def add_or_update_stock(self, code, date=None, qty=None, price=None):
        code = str(code).strip().zfill(6) # 写入时也补全
        date = date or datetime.now().strftime("%Y-%m-%d")
        qty = qty or 0
        price = price or 0.0
        
        try:
            cell = self.sheet.find(code)
            # 更新: Code(1), BuyDate(2), Qty(3), Price(4)
            self.sheet.update_cell(cell.row, 2, date)
            self.sheet.update_cell(cell.row, 3, qty)
            self.sheet.update_cell(cell.row, 4, price)
            return "Updated"
        except gspread.exceptions.CellNotFound:
            self.sheet.append_row([code, date, qty, price])
            return "Added"

    def remove_stock(self, code):
        code = str(code).strip().zfill(6)
        try:
            cell = self.sheet.find(code)
            self.sheet.delete_rows(cell.row)
            return True
        except gspread.exceptions.CellNotFound:
            return False

    def clear_all(self):
        self.sheet.resize(rows=1) 
        self.sheet.resize(rows=100)
