class KafkaReadError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)
        self.error_code = "001"


class KafkaWriteError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)
        self.error_code = "002"


class KafkaAPIErrors(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)
        self.error_code = "003"
