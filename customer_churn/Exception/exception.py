import sys
from customer_churn.Logging.logger import logger


# 2️⃣ Define Custom Exception Class
class CustomerChurnException(Exception):  # ✅ Inherits from Exception
    def __init__(self, error_message, error_details: sys):
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"Error occurred in [{self.file_name}] at line [{self.lineno}] - Message: {self.error_message}"

# 3️⃣ Exception Handling with Logging
if __name__ == '__main__':
    try:
        logger.info("Entering the Try block")
        a = 1 / 0  # Intentional division by zero error
    except Exception as e:
        logger.error("Exception occurred", exc_info=True)
        raise CustomerChurnException(e, sys)  # Raise custom exception