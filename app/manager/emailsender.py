import smtplib
from email.message import EmailMessage
import os
import json
import traceback

def send_email(recipient_email, response):
    sender_email = os.getenv("EMAIL")
    sender_password = os.getenv("EMAIL_PASSWORD")
    server_host = os.getenv("EMAIL_HOST")
    server_port = os.getenv("EMAIL_PORT")

    if not sender_email or not sender_password:
        raise ValueError("Email or password environment variables are not set.")

    # ensure the response is a string.
    if isinstance(response, dict):
        response = json.dumps(response, ensure_ascii=False, indent=2)
    if not isinstance(response, str):
        response = str(response)

    msg = EmailMessage()
    msg.set_content(response)
    msg['Subject'] = 'Student Feedback Email'
    msg['From'] = sender_email
    msg['To'] = recipient_email

    try:
        with smtplib.SMTP_SSL(server_host, server_port) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        print(f"Sending email to: {recipient_email}")
        return True

    except smtplib.SMTPException as e:
        # Log the exception details
        print(f"SMTP error occurred: {e}")
        print("Detailed error:")
        traceback.print_exc()
        return False

    except Exception as e:
        # Catch any other exceptions and log the error
        print(f"Error sending email: {e}")
        print("Detailed error:")
        traceback.print_exc()
        return False
