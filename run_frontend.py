#!/usr/bin/env python3

import http.server
import socketserver
import os
import sys

PORT = 3004

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

def main():
    frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')

    if not os.path.exists(frontend_dir):
        print("❌ Frontend directory not found!")
        print("Make sure you're running this from the SQLBoost root directory.")
        sys.exit(1)

    os.chdir(frontend_dir)

    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print("🚀 SQLBoost Frontend Server")
        print(f"📱 Frontend: http://localhost:{PORT}")
        print(f"🔗 API: http://localhost:8000")
        print(f"📚 API Docs: http://localhost:8000/docs")
        print()
        print("Make sure the API server is running: python run_api.py")
        print("Press Ctrl+C to stop the server")
        print()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Frontend server stopped")
            httpd.shutdown()

if __name__ == "__main__":
    main()