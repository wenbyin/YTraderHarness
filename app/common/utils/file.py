# import magic


# def _check_suffix(
#         raw: bytes,
#         file_name: str
# ):
#     kind = magic.from_buffer(raw[:4096], mime=True)
#
#     if not file_name.endswith(kind):
#         raise TypeError(f"file {file_name} does not end with {kind}")


def _check_empty(raw: bytes) -> None:
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    if len(raw) == 0:
        raise ValueError("empty file")


def assert_readable(
        raw: bytes,
        file_name: str
):
    # 1. 类型
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError("data must be bytes")

    # 2. 空文件
    _check_empty(
        raw=raw,
    )

    # 3. 后缀名
    # todo 前置校验

    # 4. 魔数
    # _check_suffix(
    #     raw=raw,
    #     file_name=file_name
    # )
