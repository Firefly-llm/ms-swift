# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import List, Optional, Union

import transformers
from packaging import version

transformers_ge_4_52 = version.parse(transformers.__version__) >= version.parse('4.52')


class LLMModelArch:
    qwen = 'qwen'
    llama = 'llama'
    internlm2 = 'internlm2'
    chatglm = 'chatglm'
    deepseek_v2 = 'deepseek_v2'
    baichuan = 'baichuan'

    yuan = 'yuan'
    codefuse = 'codefuse'
    phi2 = 'phi2'
    phi3 = 'phi3'
    phi3_small = 'phi3_small'
    telechat = 'telechat'
    dbrx = 'dbrx'


class MLLMModelArch:
    qwen_vl = 'qwen_vl'
    qwen_audio = 'qwen_audio'
    qwen2_vl = 'qwen2_vl'
    qwen2_audio = 'qwen2_audio'
    qwen2_5_omni = 'qwen2_5_omni'

    cogvlm = 'cogvlm'
    glm4v = 'glm4v'
    glm4_1v = 'glm4_1v'
    glm_edge_v = 'glm_edge_v'

    llama3_1_omni = 'llama3_1_omni'
    llama3_2_vision = 'llama3_2_vision'
    llama4 = 'llama4'

    llava_hf = 'llava_hf'
    llava_hf_legacy = 'llava_hf_legacy'  # transformers<4.52
    llava_next_video_hf = 'llava_next_video_hf'

    llava_llama = 'llava_llama'
    llava_mistral = 'llava_mistral'

    xcomposer = 'xcomposer'
    internvl = 'internvl'
    minicpmv = 'minicpmv'
    deepseek_vl = 'deepseek_vl'
    deepseek_vl2 = 'deepseek_vl2'
    deepseek_janus = 'deepseek_janus'

    mplug_owl2 = 'mplug_owl2'
    mplug_owl2_1 = 'mplug_owl2_1'
    mplug_owl3 = 'mplug_owl3'
    doc_owl2 = 'doc_owl2'

    phi3_vision = 'phi3_vision'
    phi4_multimodal = 'phi4_multimodal'
    florence = 'florence'
    idefics3 = 'idefics3'

    got_ocr2 = 'got_ocr2'
    dots_ocr = 'dots_ocr'

    ovis1_6 = 'ovis1_6'
    molmo = 'molmo'
    emu3_chat = 'emu3_chat'
    megrez_omni = 'megrez_omni'
    valley = 'valley'
    gemma3n = 'gemma3n'
    mistral_2503 = 'mistral_2503'
    keye_vl = 'keye_vl'

    midashenglm = 'midashenglm'


class ModelArch(LLMModelArch, MLLMModelArch):
    pass


@dataclass
class ModelKeys:
    """模型架构关键组件路径映射类
    
    类功能：
        ModelKeys 是一个数据类，用于定义和存储不同大语言模型架构中关键组件的路径映射。
        它为模型的各个部分（如 embedding、attention、mlp 等）提供统一的命名约定和访问路径，
        使得框架能够以统一的方式处理不同架构的模型组件，便于模型加载、参数访问和微调操作。
    
    继承关系说明：
        - 本类不继承其他类
        - 被 MultiModelKeys 类继承，用于多模态模型的组件映射
    
    应用场景：
        1. 模型注册：在 MODEL_ARCH_MAPPING 中注册不同模型架构的组件路径
        2. 参数访问：在模型加载、微调时，根据路径映射定位到具体的模型组件
        3. 架构适配：为不同的模型架构（如 LLaMA、Qwen、InternLM 等）提供统一的访问接口
        4. LoRA微调：在进行 LoRA 等参数高效微调时，精确定位需要微调的模块
    
    使用示例：
        >>> # 注册 LLaMA 模型架构
        >>> llama_keys = ModelKeys(
        ...     arch_name='llama',
        ...     module_list='model.layers',
        ...     mlp='model.layers.{}.mlp',
        ...     down_proj='model.layers.{}.mlp.down_proj',
        ...     attention='model.layers.{}.self_attn',
        ...     o_proj='model.layers.{}.self_attn.o_proj',
        ...     q_proj='model.layers.{}.self_attn.q_proj',
        ...     k_proj='model.layers.{}.self_attn.k_proj',
        ...     v_proj='model.layers.{}.self_attn.v_proj',
        ...     embedding='model.embed_tokens',
        ...     lm_head='lm_head',
        ... )
        >>> register_model_arch(llama_keys)
        >>> 
        >>> # 访问注册的模型架构
        >>> arch = MODEL_ARCH_MAPPING['llama']
        >>> print(arch.q_proj)  # 输出: 'model.layers.{}.self_attn.q_proj'
    
    属性说明：
    """
    
    # 模型架构名称，用于唯一标识模型架构类型（如 'llama', 'qwen', 'internlm2' 等）
    arch_name: str = None

    # 词嵌入层的路径，通常是模型的 token embedding 层（如 'model.embed_tokens'）
    embedding: str = None
    
    # 模型主干层列表的路径，指向 Transformer 层的集合（如 'model.layers'）
    module_list: str = None
    
    # 语言模型头部的路径，用于将隐藏状态映射到词表空间（如 'lm_head'）
    lm_head: str = None

    # Query 投影层的路径，注意力机制中的查询矩阵（如 'model.layers.{}.self_attn.q_proj'），其中 {} 是层索引的占位符
    q_proj: str = None
    
    # Key 投影层的路径，注意力机制中的键矩阵（如 'model.layers.{}.self_attn.k_proj'）
    k_proj: str = None
    
    # Value 投影层的路径，注意力机制中的值矩阵（如 'model.layers.{}.self_attn.v_proj'）
    v_proj: str = None
    
    # Output 投影层的路径，注意力机制的输出投影矩阵（如 'model.layers.{}.self_attn.o_proj'）
    o_proj: str = None
    
    # 注意力模块的路径，整个自注意力机制模块（如 'model.layers.{}.self_attn'）
    attention: str = None

    # MLP（多层感知机）模块的路径，前馈神经网络部分（如 'model.layers.{}.mlp'）
    mlp: str = None
    
    # MLP 下投影层的路径，前馈网络中的降维投影（如 'model.layers.{}.mlp.down_proj'）
    down_proj: str = None

    # QKV 融合投影层的路径，将 Q/K/V 三个矩阵融合为一个的投影层（如 'model.layers.{}.self_attention.query_key_value'，常见于 ChatGLM）
    qkv_proj: str = None
    
    # QK 融合投影层的路径，将 Q/K 两个矩阵融合为一个的投影层（部分特殊架构使用）
    qk_proj: str = None
    
    # Query A 投影层的路径，用于多头注意力的分组查询（部分特殊架构使用，如 DeepSeek）
    qa_proj: str = None
    
    # Query B 投影层的路径，用于多头注意力的分组查询（部分特殊架构使用，如 DeepSeek）
    qb_proj: str = None
    
    # KV 融合投影层的路径，将 K/V 两个矩阵融合为一个的投影层（如 'model.layers.{}.self_attention.key_value'，常见于 TeleChat）
    kv_proj: str = None
    
    # Key-Value A 投影层的路径，用于分组键值对注意力（部分特殊架构使用，如 DeepSeek-V2）
    kva_proj: str = None
    
    # Key-Value B 投影层的路径，用于分组键值对注意力（部分特殊架构使用，如 DeepSeek-V2）
    kvb_proj: str = None


@dataclass
class MultiModelKeys(ModelKeys):
    """多模态模型架构关键组件路径映射类
    
    类功能：
        MultiModelKeys 是 ModelKeys 的扩展类，专门用于定义和存储多模态大语言模型（如视觉-语言模型）
        架构中的关键组件路径映射。除了继承 ModelKeys 的基础组件外，还增加了多模态模型特有的组件，
        如语言模型主干、视觉编码器、模态对齐器和生成器等，使得框架能够统一处理多模态模型的各个模块。
    
    继承关系说明：
        - 继承自 ModelKeys 类，拥有基类的所有属性（embedding、attention、mlp 等）
        - 扩展了多模态模型特有的组件属性
    
    应用场景：
        1. 多模态模型注册：在 MODEL_ARCH_MAPPING 中注册视觉-语言模型架构（如 LLaVA、Qwen-VL、InternVL 等）
        2. 模态组件访问：定位和访问多模态模型的各个子模块（语言模型、视觉塔、对齐器等）
        3. 多模态微调：在进行多模态模型微调时，精确定位需要冻结或训练的特定模块
        4. 模型合并与转换：在模型格式转换或合并时，准确映射各个组件的位置
    
    使用示例：
        >>> # 注册 LLaVA 模型架构
        >>> llava_keys = MultiModelKeys(
        ...     arch_name='llava_hf',
        ...     language_model='model.language_model',
        ...     aligner='model.multi_modal_projector',
        ...     vision_tower='model.vision_tower',
        ... )
        >>> register_model_arch(llava_keys)
        >>> 
        >>> # 注册支持多个对齐器的模型
        >>> qwen_omni_keys = MultiModelKeys(
        ...     arch_name='qwen2_5_omni',
        ...     language_model='thinker.model',
        ...     vision_tower=['thinker.audio_tower', 'thinker.visual'],
        ...     aligner=['thinker.audio_tower.proj', 'thinker.visual.merger'],
        ...     generator=['talker', 'token2wav'],
        ... )
        >>> register_model_arch(qwen_omni_keys)
    """
    
    # 语言模型主干的路径，指向多模态模型中的语言模型部分（如 'model.language_model' 或 'llm'），支持单个字符串或字符串列表
    language_model: Union[str, List[str]] = field(default_factory=list)
    
    # 模态对齐器的路径，用于将视觉/音频特征映射到语言模型空间（如 'model.multi_modal_projector'），支持多个对齐器
    aligner: Union[str, List[str]] = field(default_factory=list)
    
    # 视觉/音频编码器的路径，用于提取非文本模态的特征（如 'model.vision_tower' 或 'model.audio_tower'），支持多个编码器
    # NOTE:
    # tower 并不是指字面意义上的“塔”，而是一个比喻性术语，在多模态大模型（如 Qwen-VL 等）架构中具有特定技术含义。
    # tower 在此表示 一个独立的、堆叠式的特征提取模块，通常由多层神经网络组成（如 Vision Transformer）
    # 它被称为 “tower” 是因为：
    # 1. 其结构是 层层堆叠（stacked layers） 的，形如“塔”；
    # 2. 在多模态模型中，视觉和语言通常分别由两个独立的“塔”处理（即 vision_tower 和 language_model），之后再融合。
    vision_tower: Union[str, List[str]] = field(default_factory=list)
    
    # 生成器的路径，用于生成非文本模态的输出（如图像生成器、语音合成器等），支持多个生成器
    generator: Union[str, List[str]] = field(default_factory=list)

    def __post_init__(self):
        """
        功能：
            数据类初始化后处理，统一多模态组件属性的数据类型。在 dataclass 初始化完成后自动调用，
            将多模态组件属性（language_model、aligner、vision_tower、generator）的值统一转换为列表类型。
            这样做是为了：
            1> 支持单个组件路径（字符串）和多个组件路径（列表）的统一处理
            2> 将 None 值标准化为空列表，避免后续处理时的类型检查
            3> 简化调用方代码，无需区分单组件和多组件的情况
        
        参数：
            self: MultiModelKeys 实例对象
        
        返回：
            None: 该方法直接修改实例属性，无返回值
        
        示例：
            >>> # 示例1：单个组件路径自动转换为列表
            >>> keys = MultiModelKeys(
            ...     arch_name='llava_hf',
            ...     language_model='model.language_model',  # 输入字符串
            ...     vision_tower='model.vision_tower'       # 输入字符串
            ... )
            >>> # 初始化后，自动转换为列表
            >>> print(keys.language_model)  # ['model.language_model']
            >>> print(keys.vision_tower)    # ['model.vision_tower']
            >>> 
            >>> # 示例2：多个组件路径保持列表格式
            >>> keys2 = MultiModelKeys(
            ...     arch_name='qwen2_5_omni',
            ...     vision_tower=['thinker.audio_tower', 'thinker.visual'],  # 输入列表
            ...     aligner=['thinker.audio_tower.proj', 'thinker.visual.merger']
            ... )
            >>> # 列表类型保持不变
            >>> print(keys2.vision_tower)  # ['thinker.audio_tower', 'thinker.visual']
            >>> print(keys2.aligner)        # ['thinker.audio_tower.proj', 'thinker.visual.merger']
        """
        # 遍历需要标准化的四个多模态组件属性
        for key in ['language_model', 'aligner', 'vision_tower', 'generator']:
            # 获取当前属性的值，可能是 str、List[str] 或 None
            v = getattr(self, key)
            
            # 如果属性值是字符串类型，则转换为单元素列表
            # 例如：'model.language_model' -> ['model.language_model']
            if isinstance(v, str):
                setattr(self, key, [v])
            
            # 如果属性值为 None，则初始化为空列表
            # 例如：当 aligner=None 时，转换为 aligner=[]
            # 这样后续代码可以统一使用列表操作（如 for 循环），无需额外判空
            if v is None:
                setattr(self, key, [])


MODEL_ARCH_MAPPING = {}


def register_model_arch(model_arch: ModelKeys, *, exist_ok: bool = False) -> None:
    """
    model_type: The unique ID for the model type. Models with the same model_type share
        the same architectures, template, get_function, etc.
    """
    arch_name = model_arch.arch_name
    if not exist_ok and arch_name in MODEL_ARCH_MAPPING:
        raise ValueError(f'The `{arch_name}` has already been registered in the MODEL_ARCH_MAPPING.')

    MODEL_ARCH_MAPPING[arch_name] = model_arch


register_model_arch(
    ModelKeys(
        LLMModelArch.llama,
        module_list='model.layers',
        mlp='model.layers.{}.mlp',
        down_proj='model.layers.{}.mlp.down_proj',
        attention='model.layers.{}.self_attn',
        o_proj='model.layers.{}.self_attn.o_proj',
        q_proj='model.layers.{}.self_attn.q_proj',
        k_proj='model.layers.{}.self_attn.k_proj',
        v_proj='model.layers.{}.self_attn.v_proj',
        embedding='model.embed_tokens',
        lm_head='lm_head',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.internlm2,
        module_list='model.layers',
        mlp='model.layers.{}.feed_forward',
        down_proj='model.layers.{}.feed_forward.w2',
        attention='model.layers.{}.attention',
        o_proj='model.layers.{}.attention.wo',
        qkv_proj='model.layers.{}.attention.wqkv',
        embedding='model.tok_embeddings',
        lm_head='output',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.chatglm,
        module_list='transformer.encoder.layers',
        mlp='transformer.encoder.layers.{}.mlp',
        down_proj='transformer.encoder.layers.{}.mlp.dense_4h_to_h',
        attention='transformer.encoder.layers.{}.self_attention',
        o_proj='transformer.encoder.layers.{}.self_attention.dense',
        qkv_proj='transformer.encoder.layers.{}.self_attention.query_key_value',
        embedding='transformer.embedding',
        lm_head='transformer.output_layer'))

register_model_arch(
    ModelKeys(
        LLMModelArch.telechat,
        module_list='transformer.h',
        mlp='transformer.h.{}.mlp',
        down_proj='transformer.h.{}.mlp.down_proj',
        attention='transformer.h.{}.self_attention',
        o_proj='transformer.h.{}.self_attention.dense',
        q_proj='transformer.h.{}.self_attention.query',
        kv_proj='transformer.h.{}.self_attention.key_value',
        embedding='transformer.word_embeddings',
        lm_head='lm_head'))

register_model_arch(
    ModelKeys(
        LLMModelArch.baichuan,
        module_list='model.layers',
        mlp='model.layers.{}.mlp',
        down_proj='model.layers.{}.mlp.down_proj',
        attention='model.layers.{}.self_attn',
        qkv_proj='model.layers.{}.self_attn.W_pack',
        embedding='model.embed_tokens',
        lm_head='lm_head',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.yuan,
        module_list='model.layers',
        mlp='model.layers.{}.mlp',
        down_proj='model.layers.{}.mlp.down_proj',
        attention='model.layers.{}.self_attn',
        qk_proj='model.layers.{}.self_attn.qk_proj',
        o_proj='model.layers.{}.self_attn.o_proj',
        q_proj='model.layers.{}.self_attn.q_proj',
        k_proj='model.layers.{}.self_attn.k_proj',
        v_proj='model.layers.{}.self_attn.v_proj',
        embedding='model.embed_tokens',
        lm_head='lm_head',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.codefuse,
        module_list='gpt_neox.layers',
        mlp='gpt_neox.layers.{}.mlp',
        down_proj='gpt_neox.layers.{}.mlp.dense_4h_to_h',
        attention='gpt_neox.layers.{}.attention',
        o_proj='gpt_neox.layers.{}.attention.dense',
        qkv_proj='gpt_neox.layers.{}.attention.query_key_value',
        embedding='gpt_neox.embed_in',
        lm_head='gpt_neox.embed_out',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.phi2,
        module_list='model.layers',
        mlp='model.layers.{}.mlp',
        down_proj='model.layers.{}.mlp.fc2',
        attention='model.layers.{}.self_attn',
        o_proj='model.layers.{}.self_attn.dense',
        q_proj='model.layers.{}.self_attn.q_proj',
        k_proj='model.layers.{}.self_attn.k_proj',
        v_proj='model.layers.{}.self_attn.v_proj',
        embedding='model.embed_tokens',
        lm_head='lm_head',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.qwen,
        module_list='transformer.h',
        mlp='transformer.h.{}.mlp',
        down_proj='transformer.h.{}.mlp.c_proj',
        attention='transformer.h.{}.attn',
        o_proj='transformer.h.{}.attn.c_proj',
        qkv_proj='transformer.h.{}.attn.c_attn',
        embedding='transformer.wte',
        lm_head='lm_head',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.dbrx,
        module_list='transformer.blocks',
        mlp='transformer.blocks.{}.ffn',
        attention='transformer.blocks.{}.norm_attn_norm.attn',
        o_proj='transformer.blocks.{}.norm_attn_norm.attn.out_proj',
        qkv_proj='transformer.blocks.{}.norm_attn_norm.attn.Wqkv',
        embedding='transformer.wte',
        lm_head='lm_head',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.phi3,
        module_list='model.layers',
        mlp='model.layers.{}.mlp',
        down_proj='model.layers.{}.mlp.down_proj',
        attention='model.layers.{}.self_attn',
        o_proj='model.layers.{}.self_attn.o_proj',
        qkv_proj='model.layers.{}.self_attn.qkv_proj',
        embedding='model.embed_tokens',
        lm_head='lm_head',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.phi3_small,
        module_list='model.layers',
        mlp='model.layers.{}.mlp',
        down_proj='model.layers.{}.mlp.down_proj',
        attention='model.layers.{}.self_attn',
        o_proj='model.layers.{}.self_attn.dense',
        qkv_proj='model.layers.{}.self_attn.query_key_value',
        embedding='model.embed_tokens',
        lm_head='lm_head',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.deepseek_v2,
        module_list='model.layers',
        mlp='model.layers.{}.mlp',
        down_proj='model.layers.{}.mlp.down_proj',
        attention='model.layers.{}.self_attn',
        o_proj='model.layers.{}.self_attn.o_proj',
        qa_proj='model.layers.{}.self_attn.q_a_proj',
        qb_proj='model.layers.{}.self_attn.q_b_proj',
        kva_proj='model.layers.{}.self_attn.kv_a_proj_with_mqa',
        kvb_proj='model.layers.{}.self_attn.kv_b_proj',
        embedding='model.embed_tokens',
        lm_head='lm_head',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.llava_hf_legacy,
        language_model='language_model',
        aligner='multi_modal_projector',
        vision_tower='vision_tower',
    ))

if transformers_ge_4_52:
    register_model_arch(
        MultiModelKeys(
            MLLMModelArch.llava_hf,
            language_model='model.language_model',
            aligner='model.multi_modal_projector',
            vision_tower='model.vision_tower',
        ))
else:
    register_model_arch(
        MultiModelKeys(
            MLLMModelArch.llava_hf,
            language_model='language_model',
            aligner='multi_modal_projector',
            vision_tower='vision_tower',
        ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.llava_mistral,
        language_model='model.layers',
        aligner='model.mm_projector',
        vision_tower='model.vision_tower',
    ))

if transformers_ge_4_52:
    register_model_arch(
        MultiModelKeys(
            MLLMModelArch.llava_next_video_hf,
            language_model='model.language_model',
            aligner=['model.multi_modal_projector'],
            vision_tower='model.vision_tower'))
else:
    register_model_arch(
        MultiModelKeys(
            MLLMModelArch.llava_next_video_hf,
            language_model='language_model',
            aligner=['multi_modal_projector'],
            vision_tower='vision_tower'))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.llava_llama,
        language_model='model.layers',
        aligner='model.mm_projector',
        vision_tower='model.vision_tower',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.xcomposer,
        language_model='model',
        aligner='vision_proj',
        vision_tower='vit',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.internvl,
        language_model='language_model',
        aligner='mlp1',
        vision_tower='vision_model',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.mplug_owl3,
        language_model='language_model',
        aligner='vision2text_model',
        vision_tower='vision_model',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.doc_owl2,
        language_model='model.layers',
        aligner=['model.vision2text', 'model.hr_compressor'],
        vision_tower='model.vision_model',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.deepseek_vl,
        language_model='language_model',
        aligner='aligner',
        vision_tower='vision_model',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.deepseek_janus,
        language_model='language_model',
        vision_tower='vision_model',
        aligner='aligner',
        generator=['gen_vision_model', 'gen_aligner', 'gen_head', 'gen_embed']))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.deepseek_vl2,
        language_model='language',
        vision_tower='vision',
        aligner='projector',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.minicpmv,
        language_model='llm',
        aligner='resampler',
        vision_tower='vpm',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.phi3_vision,
        language_model='model.layers',
        aligner='model.vision_embed_tokens.img_projection',
        vision_tower='model.vision_embed_tokens.img_processor',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.phi4_multimodal,
        language_model='model.layers',
        aligner=[
            'model.embed_tokens_extend.image_embed.img_projection',
            'model.embed_tokens_extend.audio_embed.audio_projection'
        ],
        vision_tower=[
            'model.embed_tokens_extend.image_embed.img_processor', 'model.embed_tokens_extend.audio_embed.encoder'
        ],
    ))

register_model_arch(MultiModelKeys(
    MLLMModelArch.cogvlm,
    language_model='model.layers',
    vision_tower='model.vision',
))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.florence,
        language_model='language_model',
        vision_tower='vision_tower',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.qwen_vl,
        language_model='transformer.h',
        vision_tower='transformer.visual',
    ))
# TODO: check lm_head, ALL
register_model_arch(
    MultiModelKeys(
        MLLMModelArch.qwen_audio,
        language_model='transformer.h',
        vision_tower='transformer.audio',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.qwen2_audio,
        language_model='language_model',
        aligner='multi_modal_projector',
        vision_tower='audio_tower',
    ))

if transformers_ge_4_52:
    register_model_arch(
        MultiModelKeys(
            MLLMModelArch.qwen2_vl,
            language_model='model.language_model',
            aligner='model.visual.merger',
            vision_tower='model.visual',
        ))
else:
    register_model_arch(
        MultiModelKeys(
            MLLMModelArch.qwen2_vl,
            language_model='model',
            aligner='visual.merger',
            vision_tower='visual',
        ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.qwen2_5_omni,
        language_model='thinker.model',
        vision_tower=['thinker.audio_tower', 'thinker.visual'],
        aligner=['thinker.audio_tower.proj', 'thinker.visual.merger'],
        generator=['talker', 'token2wav'],
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.midashenglm,
        language_model='decoder',
        aligner=['audio_projector'],
        vision_tower=['audio_encoder'],
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.glm4v,
        language_model='transformer.encoder',
        vision_tower='transformer.vision',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.glm4_1v,
        language_model='model.language_model',
        aligner='model.visual.merger',
        vision_tower='model.visual',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.idefics3,
        language_model='model.text_model',
        aligner='model.connector',
        vision_tower='model.vision_model',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.llama3_1_omni,
        language_model='model.layers',
        aligner='model.speech_projector',
        vision_tower='model.speech_encoder',
        generator='speech_generator',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.got_ocr2,
        language_model='model.layers',
        aligner='model.mm_projector_vary',
        vision_tower='model.vision_tower_high',
    ))

if transformers_ge_4_52:
    register_model_arch(
        MultiModelKeys(
            MLLMModelArch.llama3_2_vision,
            language_model='model.language_model',
            aligner='model.multi_modal_projector',
            vision_tower='model.vision_model',
        ))
else:
    register_model_arch(
        MultiModelKeys(
            MLLMModelArch.llama3_2_vision,
            language_model='language_model',
            aligner='multi_modal_projector',
            vision_tower='vision_model',
        ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.llama4,
        language_model='language_model',
        aligner='multi_modal_projector',
        vision_tower='vision_model',
    ))

register_model_arch(MultiModelKeys(
    MLLMModelArch.ovis1_6,
    language_model='llm',
    vision_tower='visual_tokenizer',
))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.molmo,
        language_model='model.transformer',
        vision_tower='model.vision_backbone',
        aligner='model.vision_backbone.image_projector'))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.megrez_omni,
        language_model='llm',
        vision_tower=['vision', 'audio'],
    ))

register_model_arch(MultiModelKeys(MLLMModelArch.emu3_chat, language_model='model'))

register_model_arch(
    MultiModelKeys(MLLMModelArch.glm_edge_v, language_model='model.layers', vision_tower='model.vision'))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.valley,
        language_model='model',
        vision_tower=['model.vision_tower', 'model.qwen2vl_vision_tower'],
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.gemma3n,
        language_model='model.language_model',
        aligner=['model.embed_vision', 'model.embed_audio'],
        vision_tower=['model.vision_tower', 'model.audio_tower'],
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.keye_vl,
        language_model='model',
        aligner='mlp_AR',
        vision_tower='visual',
    ))

register_model_arch(MultiModelKeys(
    MLLMModelArch.dots_ocr,
    language_model='model',
))


def get_model_arch(arch_name: Optional[str]) -> Optional[MultiModelKeys]:
    return MODEL_ARCH_MAPPING.get(arch_name)
