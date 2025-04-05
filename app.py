import gradio as gr
from typing import List, Dict, Tuple
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# 定义三元组类型
Triple = Tuple[str, str, str]

model_path = "YOUR_MODEL_PATH_HERE"
device = "cuda:0"


class HallucinationVerifier:
    def __init__(self):
        """初始化幻觉验证器"""

        # 使用torch.bfloat16以更好的性能
        print(f"正在加载模型 {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            # trust_remote_code=True
        )

        print(f"模型已加载到设备: {device}")

    def call_llm(self, messages):
        """使用本地模型生成回复"""
        try:
            system_prompt = ""
            prompt = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role == "system":
                    system_prompt = content
                else:
                    prompt = content

            # 使用transformers生成回复
            inputs = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)

            generate_ids = self.model.generate(
                inputs,
                max_new_tokens=800,  # 限制新生成token数量
                do_sample=False,  # 关闭随机采样
                eos_token_id=self.tokenizer.eos_token_id,  # 设置结束符
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=0,  # 关闭随机性
            )

            response = self.tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            print(f"raw response: {response}")
            return response
        except Exception as e:
            return f"模型调用错误: {str(e)}"

    def extract_json_from_markdown(self, text: str):
        """提取带markdown代码块标识的JSON"""
        # 使用非捕获组处理可能的语法变体 (允许```json或```)
        pattern = r"(?:```json?)?\n(.*?)\n```"

        # 开启多行和点匹配模式
        matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)

        # 逆序检查所有匹配项
        for candidate in reversed(matches):
            try:
                # 清理前后空格和换行
                cleaned = candidate.strip()

                # 验证JSON格式和必要字段
                result = json.loads(cleaned)
                return result
            except Exception as e:
                continue

        # 添加容错机制：当无法提取时返回错误标识
        return {"error": "JSON_NOT_FOUND"}

    def agent1_verify(self, triple: Triple) -> Dict:
        """第一个智能体：初步检查实体是否存在幻觉"""
        head, relation, tail = triple

        system_prompt = """
        你是一个专门验证知识三元组的AI助手。你的任务是确定三元组中的实体是否是真实存在的，
        而不是评估关系的正确性。请检查头实体和尾实体是否都是真实世界中存在的概念或对象。
        
        回复格式要求为JSON:
        {
            "head_entity_real": true/false,
            "head_entity_reason": "判断理由",
            "tail_entity_real": true/false,
            "tail_entity_reason": "判断理由",
            "analysis": "总体分析"
        }
        
        请直接返回JSON格式，不要添加Markdown代码块或其他格式。
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"请验证以下三元组中实体是否真实存在：({head}, {relation}, {tail})",
            },
        ]

        response = self.call_llm(messages)
        try:
            # 先处理可能存在的Markdown格式
            json_content = self.extract_json_from_markdown(response)
            print(f"agent1's response: {json_content}")
            return json_content
        except:
            # 如果返回的不是有效JSON，尝试格式化
            return {
                "head_entity_real": None,
                "head_entity_reason": "解析错误",
                "tail_entity_real": None,
                "tail_entity_reason": "解析错误",
                "analysis": response,
            }

    def agent2_verify(self, triple: Triple, agent1_result: Dict) -> Dict:
        """第二个智能体：基于第一个智能体的结果进行深度事实核查"""
        head, relation, tail = triple

        system_prompt = """
        你是一个专门进行事实核查的AI助手。前一个智能体已经对三元组中的实体进行了初步验证，
        现在你需要对这个结果进行更深入的分析。特别关注可能的幻觉实体，并进行事实核查。
        
        请重点考虑:
        1. 实体在特定领域中的存在性
        2. 实体名称是否有歧义
        3. 实体是否可能是虚构的、错误的或不存在的
        
        回复格式要求为JSON:
        {
            "head_entity_verification": {
                "is_real": true/false,
                "confidence": "高/中/低",
                "evidence": "支持你判断的证据"
            },
            "tail_entity_verification": {
                "is_real": true/false,
                "confidence": "高/中/低", 
                "evidence": "支持你判断的证据"
            },
            "relationship_validity": "对实体间关系的评估",
            "detailed_analysis": "详细分析"
        }
        """

        # 将第一个智能体的结果格式化为字符串
        agent1_result_str = json.dumps(agent1_result, ensure_ascii=False, indent=2)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
            三元组: ({head}, {relation}, {tail})
            
            第一个智能体的验证结果:
            {agent1_result_str}
            
            请你进行更深入的事实核查，并给出你的判断。
            """,
            },
        ]

        response = self.call_llm(messages)
        try:
            # 先处理可能存在的Markdown格式
            json_content = self.extract_json_from_markdown(response)
            return json_content
        except:
            return {
                "head_entity_verification": {
                    "is_real": None,
                    "confidence": "低",
                    "evidence": "解析错误",
                },
                "tail_entity_verification": {
                    "is_real": None,
                    "confidence": "低",
                    "evidence": "解析错误",
                },
                "relationship_validity": "无法确定",
                "detailed_analysis": response,
            }

    def agent3_verify(
        self, triple: Triple, agent1_result: Dict, agent2_result: Dict
    ) -> Dict:
        """第三个智能体：整合前两个智能体的结果，给出最终判断和修改建议"""
        head, relation, tail = triple

        system_prompt = """
        你是一个决策者AI助手。两个前置智能体已经对三元组中的实体进行了验证和事实核查。
        现在，你需要整合这些信息，给出最终判断，并在必要时提供修改建议。
        
        如果发现幻觉实体，请推荐替换为真实、合适的实体。
        
        回复格式要求为JSON:
        {
            "final_judgment": {
                "head_entity_hallucination": true/false,
                "tail_entity_hallucination": true/false,
                "relationship_valid": true/false
            },
            "correction_suggestion": {
                "corrected_triple": ["修正后的头实体", "修正后的关系", "修正后的尾实体"],
                "confidence": "高/中/低",
                "explanation": "修正建议的解释"
            },
            "reasoning": "你的推理过程",
            "summary": "简短总结"
        }
        """

        agent1_result_str = json.dumps(agent1_result, ensure_ascii=False, indent=2)
        agent2_result_str = json.dumps(agent2_result, ensure_ascii=False, indent=2)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
            原始三元组: ({head}, {relation}, {tail})
            
            智能体1验证结果:
            {agent1_result_str}
            
            智能体2验证结果:
            {agent2_result_str}
            
            请整合以上信息，给出最终判断和修改建议。
            """,
            },
        ]

        response = self.call_llm(messages)
        try:
            # 先处理可能存在的Markdown格式
            json_content = self.extract_json_from_markdown(response)
            return json_content
        except:
            return {
                "final_judgment": {
                    "head_entity_hallucination": None,
                    "tail_entity_hallucination": None,
                    "relationship_valid": None,
                },
                "correction_suggestion": {
                    "corrected_triple": [head, relation, tail],
                    "confidence": "低",
                    "explanation": "无法解析第三个智能体的结果",
                },
                "reasoning": "解析错误",
                "summary": response,
            }

    def full_verification(self, triple: Triple) -> Tuple[Dict, Dict, Dict]:
        """执行完整的三步验证流程"""
        agent1_result = self.agent1_verify(triple)
        agent2_result = self.agent2_verify(triple, agent1_result)
        agent3_result = self.agent3_verify(triple, agent1_result, agent2_result)

        return agent1_result, agent2_result, agent3_result


def create_gradio_interface():
    verifier = HallucinationVerifier()

    # 自定义CSS样式
    custom_css = """
    .container {
        max-width: 1000px;
        margin: auto;
    }
    .title {
        text-align: center;
        color: #1a5276;
        font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 12px;  /* 从25px减小到12px */
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(to right, #2980b9, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0 5px 0;  /* 减少底部内边距 */
        border-bottom: 2px solid #e8f4fc;
    }
    .subtitle {
        text-align: center;
        color: #34495E;
        font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
        font-size: 1.1em;  /* 稍微减小字体 */
        margin-bottom: 10px;  /* 从20px减小到10px */
        font-style: italic;
        letter-spacing: 0.5px;
        line-height: 1.2;  /* 从1.4减小到1.2 */
    }
    .input-container, .output-container {
        padding: 18px;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        margin-bottom: 16px;
        position: relative;
        overflow: hidden;
    }
    .input-container {
        background: linear-gradient(to bottom right, #f9fafc, #f4f7fa);
        border: 1px solid #e1e8ed;
    }
    .input-container:before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(to bottom, #3498db, #2980b9);
    }
    .output-container {
        background: linear-gradient(to bottom right, #f0f5fa, #e8f0f8);
        border: 1px solid #d8e2ef;
    }
    .output-container:before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(to bottom, #27ae60, #1e8449);
    }
    .input-container:hover, .output-container:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .input-container h3, .output-container h3 {
        color: #2c3e50;
        font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
        margin-bottom: 15px;
        font-weight: 600;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(0,0,0,0.05);
    }
    .verify-button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .verify-button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .agent-output {
        font-family: 'Consolas', 'Source Code Pro', monospace;
        border-left: 3px solid #3498db;
        padding-left: 10px;
    }
    .result-highlight {
        background-color: #e8f4fc;
        border-left: 3px solid #27ae60;
        padding: 10px;
        border-radius: 5px;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .fade-in {
        animation: fadeIn 0.5s ease-in-out;
    }
    """

    def process_verification(head_entity, relation, tail_entity):
        triple = (head_entity, relation, tail_entity)
        agent1_result, agent2_result, agent3_result = verifier.full_verification(triple)

        # 美化JSON显示
        agent1_str = json.dumps(agent1_result, ensure_ascii=False, indent=2)
        agent2_str = json.dumps(agent2_result, ensure_ascii=False, indent=2)
        agent3_str = json.dumps(agent3_result, ensure_ascii=False, indent=2)

        # 提取修正后的三元组
        try:
            corrected_head = agent3_result["correction_suggestion"]["corrected_triple"][
                0
            ]
            corrected_relation = agent3_result["correction_suggestion"][
                "corrected_triple"
            ][1]
            corrected_tail = agent3_result["correction_suggestion"]["corrected_triple"][
                2
            ]
        except:
            corrected_head = head_entity
            corrected_relation = relation
            corrected_tail = tail_entity

        # 添加简单的延迟以显示加载效果
        import time

        time.sleep(0.5)

        return (
            agent1_str,
            agent2_str,
            agent3_str,
            corrected_head,
            corrected_relation,
            corrected_tail,
        )

    # 创建一个加载指示动画的函数
    def show_loading():
        return ["处理中...", "处理中...", "处理中...", "", "", ""]

    with gr.Blocks(
        title="多智能体校验系统", css=custom_css, theme=gr.themes.Soft()
    ) as demo:
        with gr.Row(elem_classes="container"):
            with gr.Column():
                gr.Markdown("# 多智能体校验系统", elem_classes="title")
                gr.Markdown(
                    "输入一个三元组(头实体, 关系, 尾实体)，系统将使用多个智能体进行校验。",
                    elem_classes="subtitle",
                )

                with gr.Group(elem_classes="input-container fade-in"):
                    gr.Markdown(" ### 输入三元组")
                    with gr.Row():
                        head_entity = gr.Textbox(
                            label="头实体", placeholder="例如：爱因斯坦"
                        )
                        relation = gr.Textbox(label="关系", placeholder="例如：发明了")
                        tail_entity = gr.Textbox(
                            label="尾实体", placeholder="例如：相对论"
                        )

                    verify_button = gr.Button("开始校验", elem_classes="verify-button")

                with gr.Group(elem_classes="output-container fade-in"):
                    gr.Markdown(" ### 修正结果")
                    with gr.Row():
                        corrected_head = gr.Textbox(
                            label="修正后的头实体", elem_classes="result-highlight"
                        )
                        corrected_relation = gr.Textbox(
                            label="修正后的关系", elem_classes="result-highlight"
                        )
                        corrected_tail = gr.Textbox(
                            label="修正后的尾实体", elem_classes="result-highlight"
                        )

                with gr.Accordion("校验过程详情", open=False, elem_classes="fade-in"):
                    with gr.Tab("智能体1: 初步验证"):
                        agent1_output = gr.JSON(elem_classes="agent-output")
                    with gr.Tab("智能体2: 事实核查"):
                        agent2_output = gr.JSON(elem_classes="agent-output")
                    with gr.Tab("智能体3: 最终判断"):
                        agent3_output = gr.JSON(elem_classes="agent-output")

                with gr.Accordion("使用说明", open=False, elem_classes="fade-in"):
                    gr.Markdown("""
                    ### 使用说明
                    1. 在上方输入框中填写三元组的头实体、关系和尾实体
                    2. 点击"开始校验"按钮
                    3. 系统将依次调用三个智能体进行校验
                    4. 修正结果区域将显示修正后的三元组
                    5. 校验过程详情可展开查看详细的校验过程
                    
                    ### 示例输入
                    
                    **示例1：**
                    - 头实体: "爱因斯坦"
                    - 关系: "发明了"
                    - 尾实体: "相对论"
                    
                    **示例2：**
                    - 头实体: "拿破仑"
                    - 关系: "发明了"
                    - 尾实体: "蒸汽机"
                    """)

        # 设置按钮点击事件，添加加载动画效果
        loading_event = verify_button.click(
            fn=show_loading,
            outputs=[
                agent1_output,
                agent2_output,
                agent3_output,
                corrected_head,
                corrected_relation,
                corrected_tail,
            ],
        )
        loading_event.then(
            fn=process_verification,
            inputs=[head_entity, relation, tail_entity],
            outputs=[
                agent1_output,
                agent2_output,
                agent3_output,
                corrected_head,
                corrected_relation,
                corrected_tail,
            ],
        )

        # 添加示例
        gr.Examples(
            examples=[
                ["爱因斯坦", "发明了", "相对论"],
                ["拿破仑", "发明了", "蒸汽机"],
                ["马云", "创立了", "阿里巴巴"],
                ["关羽", "是", "三国演义的角色"],
            ],
            inputs=[head_entity, relation, tail_entity],
        )

    return demo


if __name__ == "__main__":
    # 创建并启动Gradio界面
    demo = create_gradio_interface()
    demo.launch()
