import csv
from datetime import datetime
from typing import List, Dict, Any, io

class FinancialDataPipeline:
    """Parses, structures, and filters enterprise transaction ledgers for targeted auditing workflows."""
    
    def __init__(self):
        pass

    def parse_csv_bytes(self, csv_content: str) -> List[Dict[str, Any]]:
        """
        Parses a raw CSV string uploaded over the network into standard structured records.
        """
        ledger = []
        # Split the string by newlines and read as CSV rows
        lines = csv_content.strip().splitlines()
        reader = csv.DictReader(lines)
        
        for row in reader:
            try:
                ledger.append({
                    "transaction_id": row["transaction_id"],
                    "timestamp": row["timestamp"],
                    "vendor": row["vendor"],
                    "amount": float(row["amount"]),
                    "metadata": row["metadata"]
                })
            except (KeyError, ValueError) as e:
                # Skip malformed headers or missing rows silently
                continue
        return ledger

    def extract_weekend_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filters an active list of records down strictly to Saturday and Sunday transactions."""
        weekend_records = []
        for txn in transactions:
            dt = datetime.strptime(txn["timestamp"], "%Y-%m-%d %H:%M:%S")
            if dt.weekday() in [5, 6]:  # 5 = Saturday, 6 = Sunday
                weekend_records.append(txn)
        return weekend_records
