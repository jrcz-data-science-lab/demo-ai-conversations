import smtplib, os
import traceback
import uuid
import json
from email.message import EmailMessage
from email_utils.email_formatter import write_email
from email_utils.pdf_generator import create_feedback_pdf
from datetime import datetime
import mimetypes

def send_email(recipient_email, response):
    sender_email = os.getenv("EMAIL")
    sender_password = os.getenv("EMAIL_PASSWORD")
    server_host = os.getenv("EMAIL_HOST")
    server_port = os.getenv("EMAIL_PORT")

    if not sender_email or not sender_password:
        raise ValueError("Email or password environment variables are not set.")

    # If response is a string, attempt to parse it as JSON
    if isinstance(response, str):
        try:
            response = json.loads(response)  # Parse the string to a dictionary
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON response - {response}")
            return False  # Handle error

    # Ensure the response is a dictionary
    if not isinstance(response, dict):
        print(f"Error: Expected 'response' to be a dictionary, got {type(response)}")
        return False

    msg = EmailMessage()
    msg['Subject'] = 'Talk2Care Feedback Resultaat'
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg.set_content(write_email(response))

    pdf_path = f"/tmp/feedback_{recipient_email.replace('@', '_')}.pdf"
    create_feedback_pdf(response, pdf_path)

    # Adding a unique Message-ID
    msg['Message-ID'] = f"<{uuid.uuid4()}@{server_host}>"

    # Adding a Date header
    msg['Date'] = datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')

    # Attach the PDF file
    try:
        # Guess the MIME type of the PDF file
        mime_type, _ = mimetypes.guess_type(pdf_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'

        # Read the file content
        with open(pdf_path, 'rb') as pdf_file:
            # Attach the file to the email with proper MIME type and name
            msg.add_attachment(pdf_file.read(), maintype='application', subtype='pdf', filename=os.path.basename(pdf_path))

        # Send the email via SMTP
        with smtplib.SMTP_SSL(server_host, server_port) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        print(f"Sending email to: {recipient_email}")

        try:
            os.remove(pdf_path)
            print(f"Deleted temporary PDF file: {pdf_path}")
        except OSError as e:
            print(f"Error removing temporary PDF file: {e}")
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
