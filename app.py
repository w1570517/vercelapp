from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello from Flask on Vercel!"

# Needed for Vercel to detect the app
if __name__ == "__main__":
    app.run()
