import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from config.settings import settings
from api.routes import router
from core.logger import setup_logger

logger = setup_logger()

def create_app() -> FastAPI:
    app_ = FastAPI(title=settings.APP_TITLE)
    
    # Static files if any (currently empty, but good to have)
    app_.mount("/static", StaticFiles(directory="ui/static"), name="static")
    
    # Include routes
    app_.include_router(router)
    
    return app_

app = create_app()

import webbrowser
from threading import Timer

if __name__ == "__main__":
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    
    # Open browser after a short delay to allow server to start
    def open_browser():
        webbrowser.open(f"http://{settings.HOST}:{settings.PORT}")
        
    Timer(1.5, open_browser).start()
    
    uvicorn.run("app:app", host=settings.HOST, port=settings.PORT, reload=True)
