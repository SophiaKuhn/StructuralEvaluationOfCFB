import asyncio
import logging
import os

from strucenglib_connect.comm_utils import websocket_send


class BrowserLogHandler(logging.FileHandler):
    def __init__(self, filename, **kwargs):
        self.loop = None
        self.openClients = set()
        logging.FileHandler.__init__(self, filename, **kwargs)
        pass

    async def _send_initial_log(self, ws):
        with open(self.baseFilename) as f:
            msgs = self.tail(f, 100)
            await websocket_send(ws, 'log-entries', msgs)

    def add_client(self, ws):
        self.openClients.add(ws)
        asyncio.create_task(self._send_initial_log(ws))

    def remove_client(self, ws):
        self.openClients.remove(ws)

    async def broadcast_log(self, entry):
        for ws in self.openClients:
            await self.send_log(ws, entry)

    async def send_log(self, ws, entry):
        await websocket_send(ws, 'log-entry', entry)

    def tail(self, f, lines=1, _buffer=4098):
        """Tail a file and get X lines from the end"""
        lines_found = []
        block_counter = -1
        while len(lines_found) < lines:
            try:
                f.seek(block_counter * _buffer, os.SEEK_END)
            except IOError:
                f.seek(0)
                lines_found = f.readlines()
                break

            lines_found = f.readlines()
            block_counter -= 1

        return lines_found[-lines:]

    def _is_ws_running(self):
        try:
            asyncio.get_running_loop()
            return True
        except:
            return False

    def emit(self, record):
        try:
            msg = self.format(record)
            if self._is_ws_running():
                asyncio.create_task(self.broadcast_log(msg))
            elif self.loop is not None:
                self.loop.create_task(self.broadcast_log(msg))

        except Exception as e:
            pass

        super().emit(record)
