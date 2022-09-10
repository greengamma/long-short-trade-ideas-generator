import streamlit as st
import streamlit_authenticator as stauth

hashed_passwords = stauth.Hasher(['123', '456', '789', '101112']).generate()
print(hashed_passwords)
