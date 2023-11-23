import asyncio
import concurrent.futures
import functools
import getopt
import http
import logging
import os
import ssl
import sys
import traceback
from contextlib import contextmanager
from io import StringIO

import websockets
from websockets.exceptions import ConnectionClosedError

from strucenglib_connect.comm_utils import websocket_receive, websocket_send
from strucenglib_connect.config import LOG_FILE_SERVER, SERIALIZE_CLIENT_TO_SERVER, SERIALIZE_SERVER_TO_CLIENT
from strucenglib_connect.serialize_pickle import serialize, \
    unserialize
from strucenglib_connect.server.browser_log import BrowserLogHandler
from strucenglib_connect.server.static_server import serve_file_request

script_dir = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger('strucenglib_server')
browserLog = BrowserLogHandler(LOG_FILE_SERVER)

WORKING_DIR = 'C:\\Temp\\'

API_PREFIX = '/api'


class WriteProxy(object):
    """
    Proxy to delegate stdout to callback
    """

    def __init__(self, orig, callback):
        self.orig = orig
        self.callback = callback

    def write(self, text):
        self.orig.write(text)
        self.callback(text)

    def __getattr__(self, attr):
        return getattr(self.orig, attr)


@contextmanager
def prefix_stdout(callback):
    current_out = sys.stdout
    try:
        sys.stdout = WriteProxy(current_out, callback)
        yield
    finally:
        sys.stdout = current_out


async def _send_error(websocket, msg):
    logger.debug('sending error: %s', msg)
    await websocket_send(websocket, 'error', msg)


async def _send_busy(websocket, msg):
    logger.debug('compute node currently busy', msg)
    await websocket_send(websocket, 'busy', msg)


async def _send_log_output(websocket, msg):
    logger.debug('trace %s', msg)
    await websocket_send(websocket, 'trace', msg)


async def _send_result(websocket, msg):
    logger.debug('analyse_and_extract_result %s', '[payload]')
    await websocket_send(websocket, 'analyse_and_extract_result', msg)


def is_compute_path(path):
    return path == f'{API_PREFIX}/compute'


def is_log_path(path):
    return path == f'{API_PREFIX}/log'


class AuthenticationLayer(websockets.WebSocketServerProtocol):
    async def process_request(self, path, headers):

        # XXX: We only require authentication for compute
        if not is_compute_path(path):
            return

        auth_header = headers.get('Authorization')
        logger.info(auth_header)
        if not auth_header:
            logger.info("Request without authentication. Aborting")
            return http.HTTPStatus.UNAUTHORIZED, [], b"Missing auth\n"


class WsServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.openComputeClients = set()
        self.computeBusy = False
        self.computeLock = asyncio.Lock()

        pass

    def serve(self, ssl=None):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        handler = functools.partial(serve_file_request, script_dir)
        start_server = websockets.serve(self.handle_client, self.host, self.port,
                                        ssl=ssl,
                                        ping_interval=None,
                                        process_request=handler)

        # for now we dont have authentication
        # create_protocol=AuthenticationLayer,
        logger.info(f'starting server {self.host}:{self.port} with tls={ssl is not None}')
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

    def _open_connection(self, ws, path):
        if is_compute_path(path):
            self.openComputeClients.add(ws)
        elif is_log_path(path):
            browserLog.add_client(ws)

    def _close_connection(self, ws, path):
        if is_compute_path(path):
            self.openComputeClients.remove(ws)
        if is_log_path(path):
            browserLog.remove_client(ws)

    async def handle_client(self, ws, path):
        if browserLog.loop is None:
            browserLog.loop = asyncio.get_running_loop()
        self._open_connection(ws, path)
        try:
            if is_log_path(path):
                await self._log(ws, path)
            elif is_compute_path(path):
                await self._compute(ws, path)
            else:
                logger.warning('unknown path: %s', path)

        except ConnectionError as e:
            pass
        except ConnectionClosedError as e:
            pass
        except Exception as e:
            try:
                msg = traceback.format_exc()
                await _send_error(ws, msg)
            except Exception as e:
                # XXX: if this fails we ignore signaling
                pass
        finally:
            self._close_connection(ws, path)

    async def _log(self, ws, path):

        async for msg in ws:
            # For now we dont do anything, we just forward log messages
            # once they are emitted
            pass

    async def _compute(self, ws, path):
        if self.computeLock.locked():
            logger.info("compute currently busy by other user. waiting...")

        async with self.computeLock:
            # logger.info("request acquired compute lock")
            method, payload = await websocket_receive(ws)
            if method is None:
                await _send_error(ws, 'no method given')
                return
            await self._compute_method_dispatch(ws, path, method, payload)

    async def _compute_method_dispatch(self, ws, path, method, payload):
        logger.info('handle request type: %s', method)

        supported_types = ['analyse_and_extract']
        if method is None or method not in supported_types:
            await _send_error(ws, 'unsuppoted message_type, supported: '
                              + str(supported_types))
            return

        if method == 'analyse_and_extract':
            await self._compute_analyze_and_extract(ws, path, method, payload)

    async def _compute_analyze_and_extract(self, ws, path, method, payload):
        execute_args = payload.get('args')
        structure_data = payload.get('structure')
        logger.info('handle request type: %s', str(execute_args))

        stdout = StringIO()

        def on_stdout_message(text):
            # Custom text format from abaqus
            text = text.replace("b''", '')
            text = text.replace("b'", '')
            text = text.replace("\\n'", '')
            text = text.replace("\\r", '')
            text = text.replace("\\n", '\n')
            text = text.strip()
            if len(text) > 0:
                # XXX: This may be very slow
                logger.info(text)
                stdout.write(text + '\n')

        structure = unserialize(structure_data, method=SERIALIZE_CLIENT_TO_SERVER)
        if structure is None:
            await _send_error(ws, 'structure is invalid. got None')
            return

        # XXX: Basic sanitzation
        filename = structure.name.replace('\\', '_').replace('/', '_').replace('..', '_')
        structure.name = os.path.basename(filename)
        structure.path = WORKING_DIR

        def run_in_thread():
            with prefix_stdout(on_stdout_message):
                structure.analyse_and_extract(**execute_args)


        success = False
        error_msg = ''
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                await asyncio.get_event_loop().run_in_executor(
                    pool, functools.partial(run_in_thread))
            success = True
        except Exception:
            error_msg = traceback.format_exc()

        if success:
            structure_data = serialize(structure, method=SERIALIZE_SERVER_TO_CLIENT)
            exec_res = {
                'stdout': stdout.getvalue(),
                'structure': structure_data,
                'structure_type': SERIALIZE_SERVER_TO_CLIENT,
            }
            await _send_result(ws, exec_res)
        else:
            await _send_error(ws, error_msg)


FORMATTER = logging.Formatter("%(asctime)s; %(levelname)s; %(message)s")


def configure_logger():
    logger = logging.getLogger('strucenglib_server')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)

    browserLog.setFormatter(FORMATTER)
    logger.addHandler(browserLog)
    logger.addHandler(console_handler)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.INFO)


def main():
    args = sys.argv[1:]
    options = "k:c:p:"
    long_options = ["key=", "cert=", "port="]
    key = None
    cert = None
    port = 8080
    try:
        arguments, values = getopt.getopt(args, options, long_options)
        for arg, val in arguments:
            if arg in ("-k", "--key"):
                key = val
            elif arg in ("-c", "--cert"):
                cert = val
            elif arg in ("-p", "--port"):
                port = val
    except getopt.error as err:
        print(str(err))

    use_ssl = (key is not None) and (cert is not None)
    ssl_context = None
    if use_ssl:
        print('using ssl', key, cert)
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

        # ssl_cert = script_dir + '/server.crt'
        # ssl_key = script_dir + '/server.key'

        ssl_context.load_cert_chain(cert, keyfile=key)

    s = WsServer('0.0.0.0', port)

    configure_logger()
    s.serve(ssl=ssl_context)


if __name__ == '__main__':
    main()
