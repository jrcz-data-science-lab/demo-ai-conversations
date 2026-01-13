import smtplib
from email.message import EmailMessage
import os
import json
import traceback
import uuid
from datetime import datetime

def send_email(recipient_email, response):
    sender_email = os.getenv("EMAIL")
    sender_password = os.getenv("EMAIL_PASSWORD")
    server_host = os.getenv("EMAIL_HOST")
    server_port = os.getenv("EMAIL_PORT")

    if not sender_email or not sender_password:
        raise ValueError("Email or password environment variables are not set.")

    # Ensure the response is a string
    if isinstance(response, dict):
        response = format_email(response)
    if not isinstance(response, str):
        response = str(response)

    msg = EmailMessage()
    msg.set_content(response)
    msg['Subject'] = 'Student Feedback Email'
    msg['From'] = sender_email
    msg['To'] = recipient_email

    # Adding a unique Message-ID
    msg['Message-ID'] = f"<{uuid.uuid4()}@{server_host}>"

    # Adding a Date header
    msg['Date'] = datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')

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

def format_email(data: dict) -> str:
    s = data["structured"]["sections"]

    return f"""
Beste student,

Bedankt voor je deelneming aan onze sessie. In deze email staan je resultaten op basis van het gesprek.

{s["summary"]}

{s["gespreksvaardigheden"]}

{s["comprehension"]}

{s["phase_feedback"]}

{s["speech"]}

{s["gordon"]}

{s["action_items"]}

{s["closing"]}

Met vriendelijke groeten,
Het Talk2Care Team
""".strip()