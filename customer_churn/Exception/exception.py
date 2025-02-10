import sys
from customer_churn.Logging.logger import logging

# 2️⃣ Define Custom Exception Class
class CustomerChurnException(Exception):  # ✅ Inherits from Exception
    def __init__(self, error_message, error_details: sys):
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"Error occurred in [{self.file_name}] at line [{self.lineno}] - Message: {self.error_message}"
