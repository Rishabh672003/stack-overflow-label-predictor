# This code adds custom REST api handler at runtime to a running Streamlit app
#

import numpy as np
import tensorflow as tf
import os, re, string, json
from tornado.web import Application, RequestHandler
from tornado.routing import Rule, PathMatches
import gc
import streamlit as st


@st.cache_resource()
def setup_api_handler(uri, handler):
    print("Setup Tornado. Should be called only once")

    # Get instance of Tornado
    tornado_app = next(
        o for o in gc.get_referrers(Application) if o.__class__ is Application
    )

    # Setup custom handler
    tornado_app.wildcard_router.rules.insert(0, Rule(PathMatches(uri), handler))


# === Usage ======
class HelloHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "content-type")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def get(self):
        self.write({"message": "hello world"})

    def post(self):
        # Get the data from the request
        data = json.loads(self.request.body)

        # Extract the 'text' field from the data
        text = data.get("text", "")

        # Write the text back in the response
        self.write({"received_text": text})


# This setup will be run only once
setup_api_handler("/api/predict", HelloHandler)
