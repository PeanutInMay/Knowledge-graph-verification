import gradio as gr
from openai import OpenAI
from typing import List, Dict, Tuple
import json
import os

# 设置OpenAI API密钥

client = OpenAI(api_key="YOUR_API_KEY_HERE", base_url="https://api.deepseek.com")

# 定义三元组类型
Triple = Tuple[str, str, str]


class HallucinationVerifier:
    def __init__(self):
        """初始化幻觉验证器"""
        # 可以在这里设置不同模型或系统提示
        self.models = {
            "agent1": "deepseek-chat",
            "agent2": "deepseek-chat",
            "agent3": "deepseek-chat",
        }

    def call_llm(self, messages: List[Dict], model: str) -> str:
        """调用OpenAI API获取回复"""
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0.1, max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"API调用错误: {str(e)}"

    def extract_json_from_markdown(self, text):
        """从可能包含Markdown代码块的文本中提取JSON内容"""
        import re

        # 尝试匹配Markdown代码块中的JSON
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(json_block_pattern, text)

        if matches:
            # 返回第一个匹配的代码块内容
            return matches[0].strip()

        # 如果没有匹配到代码块，返回原文本
        return text.strip()

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

        response = self.call_llm(messages, self.models["agent1"])
        try:
            # 先处理可能存在的Markdown格式
            json_content = self.extract_json_from_markdown(response)
            return json.loads(json_content)
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

        response = self.call_llm(messages, self.models["agent2"])
        try:
            # 先处理可能存在的Markdown格式
            json_content = self.extract_json_from_markdown(response)
            return json.loads(json_content)
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

        response = self.call_llm(messages, self.models["agent3"])
        try:
            # 先处理可能存在的Markdown格式
            json_content = self.extract_json_from_markdown(response)
            return json.loads(json_content)
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

    # 加载外部CSS文件
    css_file_path = os.path.join(os.path.dirname(__file__), "static", "style.css")

    try:
        with open(css_file_path, "r", encoding="utf-8") as f:
            custom_css = f.read()
    except Exception as e:
        print(f"无法加载CSS文件: {e}")
        # 如果文件加载失败，使用默认的空CSS
        custom_css = ""

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
