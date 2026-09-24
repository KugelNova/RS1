from flask import Flask, render_template_string, send_from_directory
import os

app = Flask(__name__)
SNAP_DIR = "/tmp/rescue_snaps"

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <title>Drone Rescue Web UI</title>
    <meta http-equiv="refresh" content="3"> <!-- Auto-refreshes every 3 seconds -->
    <style>
      body { font-family: Arial, sans-serif; background: #222; color: #fff; text-align: center; }
      .container { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; }
      .card { background: #333; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }
      img { max-width: 320px; height: auto; border: 2px solid #555; border-radius: 4px; }
    </style>
  </head>
  <body>
    <h1>🚁 Drone Thermal & RGB Intelligence UI</h1>
    <div class="container">
      {% for img in images %}
        <div class="card">
          <h3>{{ img }}</h3>
          <img src="/images/{{ img }}" alt="{{ img }}">
        </div>
      {% endfor %}
    </div>
  </body>
</html>
"""

@app.route('/')
def index():
    if not os.path.exists(SNAP_DIR):
        images = []
    else:
        images = sorted([f for f in os.listdir(SNAP_DIR) if f.endswith('.png')])
    return render_template_string(HTML_TEMPLATE, images=images)

@app.route('/images/<path:filename>')
def get_image(filename):
    return send_from_directory(SNAP_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)