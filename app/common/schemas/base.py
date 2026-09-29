from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    """
    Base model for all Document Service DTOs.

    extra="forbid":
        防止调用方/LLM 传入未定义字段。

    from_attributes=True:
        允许从 SQLAlchemy Entity 创建 Pydantic DTO。
    """

    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
    )
