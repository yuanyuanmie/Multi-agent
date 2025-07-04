import os
import time
from pathlib import Path
import mimetypes
import sys
from dataclasses import dataclass
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
# from log import LocalLogger  # 导入您的日志模块
import requests
import os
import time
from pathlib import Path
import mimetypes
import json
import logging
import sys
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Literal, AnyStr, Union, Any, List, Optional, Tuple
from typing_extensions import TypedDict
from pdf2image import convert_from_path
import shutil
import docker
# 添加项目路径
root_dir = '/home/user/users/llmporject'
sys.path.append(root_dir)
from xengine import LLM
# 添加项目路径
root_dir = '/home/user/users/llmporject/temp/Report Generator'
sys.path.append(root_dir)
from logger import LocalLogger
import re


@dataclass
class SmolDoclingConfig:
    """OCR 处理配置类"""
    api_base_url: str = "http://localhost:8001"  # API基础地址
    upload_url: str = None  # 文件上传接口（自动生成）
    convert_url: str = None  # 文档转换接口（自动生成）
    max_file_size: int = 100 * 1024 * 1024  # 100MB 文件大小限制
    allowed_extensions: tuple = ('.png', '.jpg', '.jpeg', '.gif', '.pdf')  # 允许的文件扩展名
    default_prompt: str = "Convert this document to structured Markdown format"  # 默认转换提示语
    processing_delay: float = 1.0  # 处理延迟（秒）
    llm_model: str = "gpt-4o-mini"  # 摘要提取使用的LLM模型
    log_business_name: str = "smoldocling_processor"  # 日志业务名称
    update_db: bool = False  # 是否更新数据库
    output_dir: str = "output"  # 输出目录
    temp_dir: str = "tmp"  # 临时目录
    cleanup_temp_files: bool = True

    
    def __post_init__(self):
        """初始化后自动设置URL"""
        self.upload_url = f"{self.api_base_url}/upload"
        self.convert_url = f"{self.api_base_url}/convert"
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

class SmolDoclingProcessor:
    def __init__(self, config: SmolDoclingConfig=None , task_id: Optional[int] = None):
        if config is None:
            config = SmolDoclingConfig()
        self.config = config
        self.task_id = task_id
        
        # 确保日志目录存在
        try:
            # 尝试从log模块导入LOG_DIR
            from log import LOG_DIR
            os.makedirs(LOG_DIR, exist_ok=True)
        except ImportError:
            # 如果无法导入，使用默认值
            default_log_dir = "logs"
            os.makedirs(default_log_dir, exist_ok=True)
        
        # 初始化日志系统
        self.logger = LocalLogger(update_db=config.update_db)
        self.logger.create_logger(business_name=config.log_business_name)
        
        # 初始化状态变量
        self.file_path = None
        self.file_size = None
        self.file_ext = None
        self.file_name = None
        self.markdown_content = None
        self.summary = None
        # 解析PDF的摘要
        self.summaries = []
        self.markdown_contents = []
        # 初始化LLM
        self.llm = LLM.LLMInterfaceCreator.langchain_openai(model=self.config.llm_model)
        self.output_parser = StrOutputParser()
        
        self.logger.general_message("OCR处理器初始化完成")

    def validate_file(self, file_path: str) -> bool:
        """验证文件是否符合上传要求"""
        self.file_path = file_path
        self.logger.general_message(f"开始验证文件: {file_path}")
        
        try:
            # 检查文件存在性
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 获取文件大小
            self.file_size = os.path.getsize(file_path)
            if self.file_size > self.config.max_file_size:
                raise ValueError(f"文件大小超过限制 ({self.file_size} > {self.config.max_file_size} bytes)")
            
            # 获取文件扩展名
            self.file_ext = Path(file_path).suffix.lower()
            if self.file_ext not in self.config.allowed_extensions:
                raise ValueError(f"不支持的文件类型: {self.file_ext}")
            
            # 获取文件名
            self.file_name = os.path.basename(file_path)
            
            self.logger.general_message(f"文件验证通过: {self.file_name} ({self.file_size} bytes)")
            return True
            
        except Exception as e:
            self.logger.general_warning(f"文件验证失败: {str(e)}")
            return False

    def upload_file(self) -> bool:
        """上传文件到服务器"""
        try:
            # 获取MIME类型
            mime_type, _ = mimetypes.guess_type(self.file_path)
            if not mime_type:
                mime_type = "application/octet-stream"
            
            # 准备文件数据
            files = {
                "file": (self.file_name, open(self.file_path, "rb"), mime_type)
            }
            
            self.logger.general_message(f"开始上传文件: {self.file_name}")
            
            # 发送上传请求
            response = requests.post(self.config.upload_url, files=files)
            
            # 处理响应
            if response.status_code == 201:  # HTTP_201_CREATED
                result = response.json()
                self.logger.general_message("✅ 文件上传成功!")
                self.logger.general_message(f"服务器保存路径: {result.get('saved_path')}")
                return True
            else:
                self.logger.general_warning(f"❌ 上传失败 (状态码: {response.status_code})")
                self.logger.general_warning(f"错误详情: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.general_warning(f"❌ 网络请求失败: {str(e)}")
            return False
        except Exception as e:
            self.logger.general_warning(f"❌ 发生未知错误: {str(e)}")
            return False
        finally:
            # 确保文件被关闭
            if 'files' in locals():
                files['file'][1].close()

    def convert_document(self, prompt: str = None) -> bool:
        """请求文档转换"""
        try:
            # 使用默认提示语如果未提供
            if not prompt:
                prompt = self.config.default_prompt
                
            # 准备请求数据
            payload = {
                "filename": self.file_name,
                "prompt": prompt
            }
            
            self.logger.general_message(f"请求文档转换: {self.file_name}")
            self.logger.general_message(f"转换提示: {prompt}")
            
            # 发送转换请求
            response = requests.post(
                self.config.convert_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            # 处理响应
            if response.status_code == 200:
                self.markdown_content = response.text
                self.logger.general_message("✅ 文档转换成功!")
                return True
            else:
                self.logger.general_warning(f"❌ 转换失败 (状态码: {response.status_code})")
                self.logger.general_warning(f"错误详情: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.general_warning(f"❌ 网络请求失败: {str(e)}")
            return False
        except Exception as e:
            self.logger.general_warning(f"❌ 发生未知错误: {str(e)}")
            return False

    def abstract_coreview(self) -> str:
        """从Markdown内容中提取摘要，考虑历史摘要信息"""
        try:
            if not self.markdown_content:
                self.logger.general_warning("没有可用的Markdown内容用于摘要提取")
                return None
    
            # 构建历史摘要字符串
            historical_summary = ""
            if hasattr(self, 'summaries') and self.summaries:
                historical_summary = "\n已存在的历史摘要:\n" + "\n".join(
                    [f"- {summary}" for summary in self.summaries if summary]
                )
            
            prompt_template = f'''
            Role: 论文摘要提取专家
            Profile:
                Author: AI Assistant
                Version: 1.0
                Language: 中文
                Description: 你是一个论文摘要提取专家
            Rules:
                1. 输入的信息是markdown格式的金融科技文献中的一页
                2. 从这个信息中提取最有可能是摘要的那段话，如果没有则不返回值，因为说明该页没有摘要
                3. 摘要大部分为一段话，结合上下文语义进行判断
                4. 直接返回摘要内容，不要添加任何额外文本或格式,所有的返回值都是输入信息的原文，不要编造信息
                5. 参考以下历史摘要信息:{historical_summary}
                    如果已经存在摘要则判断哪个摘要更符合真正的文献摘要（通常第一篇文献摘要最重要）
                    如果当前页面内容明显是真正的文献摘要（如标题明确、格式标准），即使存在历史摘要也返回它
                    否则，返回None避免重复摘要
            Input:
                {{content}}
            Output:
            '''
            
            # 实际应用中应该这样使用：
            prompt = PromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm | self.output_parser
            self.logger.general_message("开始提取摘要...")
            self.summary = chain.invoke({'content': self.markdown_content})

            # 清理摘要结果
            self.summary = self.summary.strip()
            if self.summary.startswith("摘要:"):
                self.summary = self.summary[3:].strip()
            
            self.logger.general_message("✅ 摘要提取成功")
            return self.summary
            
        except Exception as e:
            self.logger.general_warning(f"❌ 摘要提取失败: {str(e)}")
            return None    
            
    def process_document(self, file_path: str, prompt: str = None):
        
        """完整的文档处理流程"""
        # 任务开始
        if self.task_id:
            
            self.logger.start_task(self.config.update_db, self.task_id, "开始OCR文档处理")
        
        # 步骤1: 验证文件
        if self.task_id:
            self.logger.progress_only(self.config.update_db, self.task_id, 10)
        if not self.validate_file(file_path):
            if self.task_id:
                self.logger.generate_fail(self.config.update_db, self.task_id, 
                                        Exception("文件验证失败"))
            return None
        
        # 步骤2: 上传文件
        if self.task_id:
            self.logger.progress_only(self.config.update_db, self.task_id, 30)
        if not self.upload_file():
            if self.task_id:
                self.logger.generate_fail(self.config.update_db, self.task_id, 
                                        Exception("文件上传失败"))
            return None
        
        # 给服务器一点处理时间
        self.logger.general_message("等待服务器处理文件...")
        time.sleep(self.config.processing_delay)
        
        # 步骤3: 请求文档转换
        if self.task_id:
            self.logger.progress_only(self.config.update_db, self.task_id, 50)
        if not self.convert_document(prompt):
            if self.task_id:
                self.logger.generate_fail(self.config.update_db, self.task_id, 
                                        Exception("文档转换失败"))
            return None
        
        # 步骤4: 提取摘要
        if self.task_id:
            self.logger.progress_only(self.config.update_db, self.task_id, 70)
        summary = self.abstract_coreview()
        
        # 任务完成
        if self.task_id and summary:
            self.logger.progress_only(self.config.update_db, self.task_id, 100)
            self.summary = summary
            # self.logger.finalize_success(self.config.update_db, self.task_id, 
            #                             total_token=0,  # 实际应用中应计算token数
            #                             # report_path=summary_path
            #                             )
        
        return summary


        

    def pdf_to_images(self, pdf_path: str, pages: Optional[Tuple[int, int]] = (1, 3), dpi: int = 80) -> List[str]:
        """
        将PDF转换为高清图像并保存到临时目录
        返回图片文件路径列表
        
        :param pdf_path: PDF文件路径
        :param pages: 转换的页码范围，格式为(start, end)
        :param dpi: 转换分辨率
        :return: 生成的图片文件路径列表
        """
        self.logger.general_message(f"开始PDF转换: {pdf_path}")
        
        # 创建临时目录
        os.makedirs(self.config.temp_dir, exist_ok=True)
        
        try:
            # 计算总页数
            first_page = pages[0]
            last_page = pages[1] # if pages[1] is not None else first_page
            
            # 转换PDF为图片
            images = convert_from_path(
                pdf_path, 
                dpi=dpi,
                first_page=first_page,
                last_page=last_page,
                fmt="png"
            )
            
            # 获取保存的图片路径（使用页码号命名）
            image_paths = []
            for idx, image in enumerate(images, start=first_page):
                # 使用页面号作为文件名: page_{页码}.png
                image_name = f"page_{idx}.png"
                image_path = os.path.join(self.config.temp_dir, image_name)
                image.save(image_path)  # 确保使用明确的文件名保存
                image_paths.append(image_path)
                self.logger.general_message(f"已保存页面 {idx}: {image_path}")
            
            self.logger.general_message(f"✅ PDF转换完成，共生成 {len(image_paths)} 张图片")
            return image_paths
        
        except Exception as e:
            self.logger.general_warning(f"❌ PDF转换失败: {str(e)}")
            return []
            
    def cleanup_temp_files(self):
        """清理临时目录中的所有文件"""
        try:
            self.logger.general_message(f"开始清理临时目录: {self.config.temp_dir}")
            
            # 检查临时目录是否存在
            if os.path.exists(self.config.temp_dir) and os.path.isdir(self.config.temp_dir):
                # 遍历目录中的所有文件和子目录
                for filename in os.listdir(self.config.temp_dir):
                    file_path = os.path.join(self.config.temp_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            self.logger.general_message(f"已删除临时文件: {file_path}")
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                            self.logger.general_message(f"已删除临时子目录: {file_path}")
                    except Exception as e:
                        self.logger.general_warning(f"删除失败: {file_path}, 原因: {str(e)}")
                
                self.logger.general_message("✅ 临时文件清理完成")
            else:
                self.logger.general_message("临时目录不存在，无需清理")
            
        except Exception as e:
            self.logger.general_warning(f"清理临时文件时出错: {str(e)}")


    def safe_restart(container_name):
        try:
            container = client.containers.get(container_name)
            if container.status != 'running':
                container.restart(timeout=30)  # 设置重启超时30秒
                logging.info(f"成功重启容器: {container_name}")
        except Exception as e:
            logging.error(f"重启失败: {container_name}, 错误: {str(e)}")

    
    def extract_header_segments(self,num_sentence = 3):
        """
        params: 
            num_sentence = n: 取前n个句子
        func:
            接在process_pdf之后，因为目前只有process_pdf才会生成markdown_contents
            抽取标题及前三句话
            return： str
            
        """
        # 将process_pdf
        content_list = self.markdown_contents
        text = ''.join(content_list)
        output = ''
        results = []
        # 查找所有##开头的标题及其位置
        headers = [(m.start(), m.end(), m.group().strip()) 
                   for m in re.finditer(r'^##[^\n]*', text, re.MULTILINE)]
        
        total_len = len(text)
        
        for i, (start, end, header) in enumerate(headers):
            next_header_start = headers[i+1][0] if i+1 < len(headers) else total_len
            content_start = end + 1  # 跳过标题后的换行符
            
            # 获取标题后到下一个标题的所有内容
            content_block = text[content_start:next_header_start].strip()
            
            # 提取前三个句子
            sentences = re.split(r'(?<=[.!?])\s+', content_block)
            first_three = ' '.join(sentences[:3])  # 取前三个句子并用空格连接
            
            # 检查并确保第三个句子完整结束
            if first_three and not re.search(r'[.!?]$', first_three):
                if len(sentences) > 3:
                    first_three += sentences[3]  # 添加下一个句子直到有结束标点
                else:
                    # 尝试从最后一个句子中提取完整的部分
                    last_sentence_match = re.search(r'(.*?[.!?])', content_block)
                    if last_sentence_match:
                        first_three = last_sentence_match.group(1)
            
            # 组合标题+前三句内容
            if first_three:
                results.append(f"{header}\n\n{first_three.strip()}")
                output = ''.join(results)
        return output
    
    
    def process_pdf(self, pdf_path: str, prompt: str = None, pages: Tuple[int, int] = (1, 3), if_top3 = True):
        """
        完整的PDF处理流程
        1. 将PDF转换为图片
        2. 对每页图片执行OCR处理
        3. 收集所有页的摘要
        
        :param pdf_path: PDF文件路径
        :param prompt: 转换提示语
        :param pages: 处理的页码范围
        :return: 各页摘要列表
        """
        # 任务开始
        if self.task_id:
            safe_restart()
            self.logger.start_task(self.config.update_db, self.task_id, "开始PDF文档处理")
        
        # 步骤1: PDF转图片
        if if_top3:
            image_paths = self.pdf_to_images(pdf_path, pages=pages)
        else:
            image_paths = self.pdf_to_images(pdf_path, pages=(1,None))
        
        if not image_paths:
            if self.task_id:
                self.logger.generate_fail(self.config.update_db, self.task_id, 
                                        Exception("PDF转换失败"))
            return []
        
        # 步骤2: 逐页处理
        summaries = self.summaries
        markdown_contents = self.markdown_contents
        total_pages = len(image_paths)
        
        for idx, img_path in enumerate(image_paths):
            page_num = pages[0] + idx
            self.logger.general_message(f"🏁 开始处理页面 {page_num}/{pages[1]}")
            
            # 更新进度
            if self.task_id:
                progress = 10 + int(idx/total_pages * 90)
                self.logger.progress_only(self.config.update_db, self.task_id, progress)
            
            # 处理当前页面
            summary = self.process_document(img_path, prompt)
            markdown_contents.append(self.markdown_content)
            summaries.append(summary)
            if summary == None:
                self.logger.general_message(f"✅ 页面 {page_num} 无摘要")
            else:    
                self.logger.general_message(f"✅ 页面 {page_num} 处理完成: {summary[:50]}...")
        
        # 任务完成
        if self.task_id:
            self.logger.progress_only(self.config.update_db, self.task_id, 100)
            self.logger.finalize_success(self.config.update_db, self.task_id, 
                                       total_token=0, report_path=self.config.output_dir)
            if self.cleanup_temp_files:
                self.cleanup_temp_files()
        return summaries        
