"""Vendored from roboarena_evaluator/evaluation_client/websocket_client_policy.py.

Implements BasePolicy via a websocket connection. The server, on connect,
sends a metadata blob describing what observations it expects. Subsequent
messages discriminate between infer / reset via the `endpoint` field.
"""

import logging
from typing import Dict

import websockets.sync.client

from . import base_policy as _base_policy
from . import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self):
        logging.info(f"Waiting for server at {self._uri}...")
        try:
            conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
            metadata = msgpack_numpy.unpackb(conn.recv())
            return conn, metadata
        except Exception:
            logging.info("Connection to server with ws:// failed. Trying wss:// ...")

        self._uri = "wss://" + self._uri.split("//")[1]
        conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
        metadata = msgpack_numpy.unpackb(conn.recv())
        return conn, metadata

    def infer(self, obs: Dict) -> Dict:
        obs["endpoint"] = "infer"

        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self, reset_info: Dict) -> None:
        reset_info["endpoint"] = "reset"

        data = self._packer.pack(reset_info)
        self._ws.send(data)
        response = self._ws.recv()
        return response
