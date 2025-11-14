import uvicorn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )