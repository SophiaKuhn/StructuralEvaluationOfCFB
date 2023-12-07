from strucenglib_connect.wsclient import WsClient


def rpc_analyse_and_extract(server, data):
    c = WsClient(server)
    ws_res = c.analyse_and_extract(data)

    return {
        'status': ws_res.status,
        'stdout': ws_res.stdout,
        'payload': ws_res.payload
    }
