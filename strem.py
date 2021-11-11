import streamlit as st

from twilio.rest import Client

def sendInfo(data, mode):
    if mode == "SMS":
        from_whatsapp_number = '+16179368054'
        to_whatsapp_number = '+919812627589'
    if mode == "WhatsApp":
        from_whatsapp_number = 'whatsapp:+14155238886'
        to_whatsapp_number = 'whatsapp:+919812627589'

    account_sid = 'ACf1acf121ec795acb0f3d1440a1c9de3a'
    auth_token = 'c565bcfe6cd8ec999ee3a0e8543a9c80'
    client = Client(account_sid, auth_token)
    # index = False
    # message = client.messages.create(body=f"{data.to_csv()}", from_=from_whatsapp_number,to=to_whatsapp_number)
    client.messages.create(body=data, from_=from_whatsapp_number,to=to_whatsapp_number)

    return True

# sendInfo("data", "SMS")

# EMERGENCY,  SMS_SEND, APP_SEND = st.beta_columns(3)
# STATUS = st.empty()
#
# if EMERGENCY.button("EMERGENCY", key="a") and STATUS.info("Active"):
#     if SMS_SEND.button("SMS", key="b") and STATUS.info("SMS Sending....."):
#         STATUS.success("Done")
#     if APP_SEND.button("WhatsApp", key="c") and STATUS.info("WhatsApp Massage Sending....."):
#         STATUS.success("Done")


# EMERGENCY,  MODE, SEND = st.beta_columns(3)
STATUS = st.empty()
mode = st.sidebar.selectbox("Select MODE", ["Select","SMS", "WhatsApp"])

if (mode == "SMS") and STATUS.info("SMS Preparing....."):
    if st.sidebar.button("SEND"):
        id = sendInfo("data", "SMS")
        if id:
            STATUS.success("Done")
if (mode == "WhatsApp") and STATUS.info("WhatsApp Massage Preparing....."):
    if st.sidebar.button("SEND"):
        id = sendInfo("data", "WhatsApp")
        if id:
            STATUS.success("Done")
