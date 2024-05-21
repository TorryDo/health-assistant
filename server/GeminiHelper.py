import google.generativeai as genai


class GeminiHelper:

    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def request(self, message: str):
        return self.model.generate_content(message)
