from enum import Enum


class FileStatus(str, Enum):
    """
    File 生命周期状态。

    active:
        文件正常存在，可被 Document 引用。

    deleted:
        文件逻辑删除。
        第一阶段不立即删除物理文件。
    """

    ACTIVE = "active"
    DELETED = "deleted"


class FileOrigin(str, Enum):
    """
    File 的来源。

    user_upload:
        用户上传的文件资产。默认出现在 GET /files 列表里。

    pipeline_artifact:
        流水线中间产物（parsed_text / chunks / vectors ...）。
        系统生成，没有上传者，**默认不出现在列表里** ——
        否则文档列表旁边会混进一堆内部产物。

    拆这一列是因为产物与用户资产的**生命周期不同**：
    用户文件是资产，产物是可清理的中间态。
    将来给产物加引用计数 / 超时清理时也靠它区分。
    """

    USER_UPLOAD = "user_upload"
    PIPELINE_ARTIFACT = "pipeline_artifact"
