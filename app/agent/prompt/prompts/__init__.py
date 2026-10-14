"""
节点功能性提示词：**一个 model_role 一个文件**。

约定（四件，写在这里是因为漏掉任何一件都不会报错）
----------------------------------------------
    1. 文件名任意，但模块里必须声明一个模块级 `PROMPT`（`PromptDefinition`）
       —— 缺了它目录装配期就抛 PromptRegistryError，不是安静少一条。
    2. `PROMPT.key` 用 `ModelRole` 里的常量（那是一个**开放**词表，见
       app/agent/model/types.py：加一个角色不需要改 Model 模块）。
    3. 静态文本与动态块都留在**本文件**里：提示词文本、以及"这段运行时
       数据该怎么排版"（如能力候选清单的行格式）都是提示词的一部分，
       散到调用方去就又会回到"提示词住在节点里"的老样子。
    4. **用户消息（HumanMessage）那一半也住在本文件**，导出一个
       `render_user_message(...)`：

           PROMPT                -> 系统提示词（按 model_role 查，常量）
           render_user_message   -> 用户消息（这次的数据 + 它的排版）
           <X>Like (Protocol)    该函数需要的**最小字段集**

       ⚠️ 第 4 条是 2026-10-11 补的，而且它是**改了现状**换来的
       ------------------------------------------------
       在此之前，memory 那三份提示词的用户消息是在服务里手工拼的
       （`MemoryAtomServiceImpl._build_extract_request`、
       `_build_conflict_request`、`MemorySceneServiceImpl._build_request`）。
       代价实测出来了：表头那两句限定（"务必结合 timestamp 推算时间"、
       "只从这里提取记忆"）是**提示词的正文**，只是位置在用户消息里 ——
       放在服务里时，改提示词的人翻不到它们；而"共 N 个"这类计数在两处
       各写一遍，改一处另一处就漂。

       所以分工是：**取哪些数据**归调用方（它才知道该读哪个 scope、哪段
       水位、现在几点），**这些数据在提示词里长什么样**归本目录。字段用
       `Protocol` 声明而不是 import 领域实体（`classification.py` 对候选
       清单用了同一手法），提示词这一层因此不认识 `app.services.*`。

       ⚠️ 与提示词格式**强绑定**的解析器也跟着来
       ------------------------------
       例：`memory_scene_consolidate.py` 里的 `parse_meta` —— 它解析的
       `-----META-START-----` 块，格式就由同一个文件的模板定义。分成两处
       时，改模板忘了改解析器的症状是清单里的摘要全变空、模型从此选不准
       该 UPDATE 进哪一个，而**不报错**。

⚠️ 本目录必须有 `__init__.py`
---------------------------
目录装配靠 `pkgutil.walk_packages`（见 app/common/utils/reflection.py），
它**不会走进没有 `__init__.py` 的目录**。删掉这个文件不会报错 ——
所有提示词会无声消失，症状是"每个 model_role 都 PromptNotFoundError"。
app/agent/tools/processor/__init__.py 已经为同一件事留过警告。

加一份提示词 = 加一个文件，不用回来改任何清单。
"""
