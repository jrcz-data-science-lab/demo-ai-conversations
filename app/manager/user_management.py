import re
from db_utils import get_or_create_user

def ensure_user(username):
    """
    Makes sure a user exists in the database.
    Returns the user's ID.
    """
    return get_or_create_user(username)

def validation(username):
    """
    Validates the username to ensure it's a valid email address
    that ends with @hz.nl and contains both letters and numbers.
    """
    # Check if the email is valid and ends with @hz.nl
    email_regex = r'^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$'  # Basic email pattern

    if re.match(email_regex, username) and username.endswith('@hz.nl'):
        local_part = username.split('@')[0]  # Part before the @
        if any(char.isdigit() for char in local_part) and any(char.isalpha() for char in local_part):
            return True  # Valid email with both letters and numbers
        else:
            return False  # Does not contain both letters and numbers
    return False  # Not a valid email or does not end with @hz.nl
