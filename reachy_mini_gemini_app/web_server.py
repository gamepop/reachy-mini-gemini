"""Web server for Reachy Mini Gemini App settings page."""

import json
import logging
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from reachy_mini_gemini_app.config import get_settings_for_api, save_settings

logger = logging.getLogger(__name__)

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"

# Default port for the settings server
DEFAULT_PORT = 8042


class SettingsHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the settings API and static files."""

    def __init__(self, *args, **kwargs):
        # Set the directory to serve static files from
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.debug(f"HTTP: {format % args}")

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)

        if parsed.path == "/api/settings":
            self._handle_get_settings()
        elif parsed.path == "/api/health":
            self._send_json({"status": "ok"})
        else:
            # Serve static files
            super().do_GET()

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)

        if parsed.path == "/api/settings":
            self._handle_post_settings()
        else:
            self.send_error(404, "Not Found")

    def _handle_get_settings(self):
        """Return current settings."""
        settings = get_settings_for_api()
        self._send_json(settings)

    def _handle_post_settings(self):
        """Save new settings."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            new_settings = json.loads(body.decode("utf-8"))

            # Remove undefined/null values
            new_settings = {k: v for k, v in new_settings.items() if v is not None}

            if save_settings(new_settings):
                self._send_json({"success": True})
            else:
                self.send_error(500, "Failed to save settings")
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            self.send_error(500, str(e))

    def _send_json(self, data: dict):
        """Send a JSON response."""
        response = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response)

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


class SettingsServer:
    """Settings web server that runs in a background thread."""

    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> str:
        """Start the settings server in a background thread. Returns the URL."""
        if self.server is not None:
            return self.get_url()

        try:
            self.server = HTTPServer(("0.0.0.0", self.port), SettingsHandler)
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info(f"Settings server started at {self.get_url()}")
            return self.get_url()
        except OSError as e:
            logger.error(f"Failed to start settings server on port {self.port}: {e}")
            raise

    def _run(self):
        """Run the server (called in background thread)."""
        if self.server:
            self.server.serve_forever()

    def stop(self):
        """Stop the settings server."""
        if self.server:
            self.server.shutdown()
            self.server = None
            self.thread = None
            logger.info("Settings server stopped")

    def get_url(self) -> str:
        """Get the server URL."""
        return f"http://0.0.0.0:{self.port}"


# Global server instance
_server: Optional[SettingsServer] = None


def start_settings_server(port: int = DEFAULT_PORT) -> str:
    """Start the global settings server."""
    global _server
    if _server is None:
        _server = SettingsServer(port)
    return _server.start()


def stop_settings_server():
    """Stop the global settings server."""
    global _server
    if _server:
        _server.stop()
        _server = None
