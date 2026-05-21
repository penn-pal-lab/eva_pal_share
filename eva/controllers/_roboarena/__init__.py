"""Vendored from https://github.com/pranavatreya/roboarena_evaluator (MIT).

These modules implement the RoboArena policy-server protocol (msgpack-numpy
serialization, websocket transport with `endpoint` dispatch, server-driven
metadata). Vendored so the EVA controller can talk to RoboArena-compliant
servers without taking a runtime dependency on the upstream repo.
"""
