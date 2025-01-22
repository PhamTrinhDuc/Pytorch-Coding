import time
import sys
from pathlib import Path
from typing import Callable
sys.path.append(str(Path(__file__).parent.parent))

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from logs.logger import Logger

LOGGER = Logger(
    name=__file__, 
    log_file="http.log"
)

class LogProcessAndTime(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, 
                       call_next: Callable):
        st_time = time.time()
        response = await call_next(request)
        process_time = time.time() - st_time

        LOGGER.log.info(msg=
                        f"{request.client.host} - " 
                        f"{request.method} - " 
                        f"{request.url.path} - " 
                        f"{request.scope['http_version']} - "
                        f"{response.status_code} - " 
                        f"{process_time:.2f}s"
        )
        return response