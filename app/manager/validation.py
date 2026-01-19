import re

def validate_inputs(username, scenario):
    if not scenario:
        return {"validation": False, "errormessage": "No scenario specified."}
    if not username:
        return {"validation": False, "errormessage": "Email is required."}
    
    # Step 2: Auto-fill @hz.nl if missing
    if '@hz.nl' not in username:
        username = username + '@hz.nl'
    
    # Step 3: Validate email format using regex
    email_pattern = r"^[a-zA-Z0-9._%+-]+@hz\.nl$"
    if not re.match(email_pattern, username):
        return {"validation": False, "errormessage": "Invalid email address. Must be a valid @hz.nl email."}
    
    # If all checks pass
    return {"validation": True, "errormessage": ""}