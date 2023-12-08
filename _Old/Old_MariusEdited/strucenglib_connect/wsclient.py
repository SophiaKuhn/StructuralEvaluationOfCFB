import asyncio
import io
import logging
import ssl
import traceback

import requests
import websockets
from requests.exceptions import SSLError

from strucenglib_connect.comm_utils import websocket_receive, websocket_send

logger = logging.getLogger(__name__)


class WsResult:
    def __init__(self):
        self.status = 'error'
        self.stdout = ''
        self.payload = None


class WsClient:
    def __init__(self, host):
        self.host = host
        self.is_alive = False
        pass

    async def check_alive(self):
        while self.is_alive:
            print('server is alive')
            await asyncio.sleep(20)

    def host_online(self):
        def check_url(url, prefix):
            url_parts = url.split('://')
            if len(url_parts) > 1:
                url = prefix + url_parts[1]
            try:
                requests.head(url, timeout=8)
                return True
            except SSLError as e:
                return True
            except:
                return False

        return check_url(self.host, 'http://') or check_url(self.host, 'https://')

    def analyse_and_extract(self, payload):
        stdout_buffer = io.StringIO()
        result = WsResult()

        if not self.host_online():
            result.stdout = 'Host ' + self.host + ' cannot be reached. Check connectivity.'
            result.status = 'error'
            return result

        asyncio.get_event_loop().run_until_complete(asyncio.wait([
            self.check_alive(),
            self.async_processing(payload, stdout_buffer, result)
        ]))
        result.stdout = stdout_buffer.getvalue()
        stdout_buffer.close()
        return result

    def _is_secure_url(self, url):
        return url.startswith('wss')

    async def async_processing(self, payload, stdout_buffer, result):
        def _do_print(msg):
            print('do_print: ', msg)
            msg = msg.replace('\\r\\n', '\n')
            msg = msg.replace('\\n', '\n')
            stdout_buffer.write(msg)

        try:
            ctx = None
            if self._is_secure_url(self.host):
                # XXX: We use self signed certs so ignore cert errors
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            async with websockets.connect(self.host,
                                          ping_interval=None,
                                          ssl=ctx) as ws:
                await websocket_send(ws, 'analyse_and_extract', payload)
                while True:
                    method, payload = await websocket_receive(ws)

                    if method == 'trace':
                        _do_print(payload)

                    elif method == 'analyse_and_extract_result':
                        result.status = 'success'
                        result.payload = payload.get('structure')
                        _do_print(payload.get('stdout'))
                        break

                    elif method == 'error':
                        _do_print('Error from server: ' + payload)
                        break
                    else:
                        _do_print('unknown type:' + method + ', payload: ' + payload)


        except Exception as e:
            error_msg = traceback.format_exc()
            _do_print(error_msg)

        self.is_alive = False
