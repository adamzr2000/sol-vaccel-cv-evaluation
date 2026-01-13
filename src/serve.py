import uvicorn

if __name__ == "__main__":
    # Runs the app module with reload enabled for development
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)