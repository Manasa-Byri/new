import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=["app"],   # only watch app/ — ignore scripts at project root
        log_level="info"
    )
