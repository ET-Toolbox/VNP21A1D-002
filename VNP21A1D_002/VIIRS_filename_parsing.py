from os.path import basename
from datetime import date, datetime

def parse_VIIRS_product(filename: str) -> str:
    """
    Extracts the product name from a VIIRS filename.

    Args:
        filename (str): The VIIRS filename.

    Returns:
        str: The product name extracted from the filename.
    """
    # Split the filename by '.' and return the first part as the product name
    return str(basename(filename).split(".")[0])

def parse_VIIRS_date(filename: str) -> date:
    """
    Extracts the date from a VIIRS filename and converts it to a date object.

    Args:
        filename (str): The VIIRS filename.

    Returns:
        date: The date extracted from the filename.
    """
    # Split the filename by '.' and parse the second part as a date in the format YYYYDDD
    return datetime.strptime(basename(filename).split(".")[1][1:], "%Y%j").date()

def parse_VIIRS_tile(filename: str) -> str:
    """
    Extracts the tile identifier from a VIIRS filename.

    Args:
        filename (str): The VIIRS filename.

    Returns:
        str: The tile identifier extracted from the filename.
    """
    # Split the filename by '.' and return the third part as the tile identifier
    return str(basename(filename).split(".")[2])

def parse_VIIRS_build(filename: str) -> int:
    """
    Extracts the build number from a VIIRS filename and converts it to an integer.

    Args:
        filename (str): The VIIRS filename.

    Returns:
        int: The build number extracted from the filename.
    """
    # Split the filename by '.' and return the fourth part as the build number
    return int(basename(filename).split(".")[3])