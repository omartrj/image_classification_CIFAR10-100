import logging
from rich.logging import RichHandler

def get_logger(name: str = "app", level: int = logging.DEBUG) -> logging.Logger:
    # Configura il formato del logger
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True)]
    )
    # Restituisce il logger con il nome specificato
    return logging.getLogger(name)
