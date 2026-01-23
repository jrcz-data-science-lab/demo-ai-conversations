import re
from db_utils import get_or_create_user

def ensure_user(username):
    """
    Makes sure a user exists in the database.
    Returns the user's ID.
    """
    return get_or_create_user(username)