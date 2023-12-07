# strucenglib_connect

Connection layer to execute functionality of compas fea on a remote server.

To prevent remote code execution, server accepts a json datastructure of compas fea python objects whose class definitions are looked up on the server.
Data can dynamically be serialized and sent to server but solely data and type definitions no code is sent.
The receiver has to have the same python packages available in environment. An optional white listing allows to white list packages to reconstruct.
